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

// test
#include <catch.hpp>

// sirius
#include <cuda/vss/cudf_raft_interop.hpp>

#include <../../../src/cuda/vss/brute_force_search.hpp>

// cudf
#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>

// raft / rmm / cuvs
#include <raft/core/device_mdspan.hpp>
#include <raft/core/device_resources.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>

#include <cuda/stream>
#include <cuda_runtime_api.h>

#include <cuda_fp16.h>
#include <cuvs/distance/distance.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace {

using sirius::vss::brute_force_knn;
using sirius::vss::dataset_matrix_view;
using sirius::vss::list_column_as_dataset_view;
using Metric = cuvs::distance::DistanceType;

// Build a Sirius-style ARRAY<FLOAT>[dim] column (cudf LIST with a contiguous,
// uniform FLOAT32 values child) so list_column_as_dataset_view can wrap it.
std::unique_ptr<cudf::column> make_fixed_size_float_list(std::vector<float> const& values,
                                                         cudf::size_type n_rows,
                                                         cudf::size_type dim)
{
  auto child = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::FLOAT32}, n_rows * dim, cudf::mask_state::UNALLOCATED);
  cudaMemcpy(child->mutable_view().data<float>(),
             values.data(),
             sizeof(float) * values.size(),
             cudaMemcpyHostToDevice);

  std::vector<int32_t> offsets(n_rows + 1);
  for (cudf::size_type i = 0; i <= n_rows; ++i) {
    offsets[i] = i * dim;
  }
  auto offsets_col = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::INT32}, n_rows + 1, cudf::mask_state::UNALLOCATED);
  cudaMemcpy(offsets_col->mutable_view().data<int32_t>(),
             offsets.data(),
             sizeof(int32_t) * offsets.size(),
             cudaMemcpyHostToDevice);

  return cudf::make_lists_column(
    n_rows, std::move(offsets_col), std::move(child), 0, rmm::device_buffer{});
}

// Copy a device column's raw elements back to the host after the search stream
// has been synchronized.
template <typename T>
std::vector<T> to_host(cudf::column_view const& col)
{
  std::vector<T> out(col.size());
  cudaMemcpy(out.data(), col.data<T>(), sizeof(T) * out.size(), cudaMemcpyDeviceToHost);
  return out;
}

}  // namespace

TEST_CASE("brute_force_knn returns the exact nearest rows in order", "[vss]")
{
  ::cuda::stream_ref stream = cudf::get_default_stream();
  raft::device_resources res{stream};

  // Dataset row i is [i, i, i]; query is the origin, so distances grow with i and
  // the nearest rows are the smallest ids, tie-free.
  constexpr cudf::size_type n_rows = 8;
  constexpr cudf::size_type dim    = 3;
  std::vector<float> data(n_rows * dim);
  for (cudf::size_type i = 0; i < n_rows; ++i) {
    data[i * dim + 0] = static_cast<float>(i);
    data[i * dim + 1] = static_cast<float>(i);
    data[i * dim + 2] = static_cast<float>(i);
  }
  auto dataset_col  = make_fixed_size_float_list(data, n_rows, dim);
  auto dataset_view = list_column_as_dataset_view(dataset_col->view(), dim);

  std::vector<float> q_host(dim, 0.0f);
  rmm::device_uvector<float> q_dev(dim, stream);
  cudaMemcpyAsync(
    q_dev.data(), q_host.data(), sizeof(float) * dim, cudaMemcpyHostToDevice, stream.get());
  auto query_view =
    raft::make_device_matrix_view<const float, int64_t, raft::row_major>(q_dev.data(), 1, dim);

  constexpr int64_t k = 3;
  auto knn            = brute_force_knn(res, dataset_view, query_view, k, Metric::L2SqrtUnexpanded);
  stream.sync();

  REQUIRE(knn.n_queries == 1);
  REQUIRE(knn.k == k);
  REQUIRE(knn.neighbors->size() == k);
  REQUIRE(knn.distances->size() == k);

  auto neighbors = to_host<int64_t>(knn.neighbors->view());
  auto distances = to_host<float>(knn.distances->view());

  // Nearest-first: rows 0, 1, 2 with Euclidean distance i * sqrt(3).
  REQUIRE(neighbors == std::vector<int64_t>{0, 1, 2});
  for (int64_t i = 0; i < k; ++i) {
    REQUIRE(distances[i] == Approx(static_cast<float>(i) * std::sqrt(3.0f)).margin(1e-3));
  }
}

TEST_CASE("brute_force_knn cosine orders by angle, not magnitude", "[vss]")
{
  ::cuda::stream_ref stream = cudf::get_default_stream();
  raft::device_resources res{stream};

  constexpr cudf::size_type dim = 2;
  // Row 0 points along +x (aligned with the query), row 1 is off-axis, row 2 is
  // aligned but far in magnitude. Cosine distance ignores magnitude, so the two
  // aligned rows tie at 0 and both beat the off-axis row.
  std::vector<float> data{1.0f,
                          0.0f,  // row 0: aligned
                          1.0f,
                          1.0f,  // row 1: 45 deg
                          9.0f,
                          0.0f};  // row 2: aligned, large magnitude
  constexpr cudf::size_type n_rows = 3;
  auto dataset_col                 = make_fixed_size_float_list(data, n_rows, dim);
  auto dataset_view                = list_column_as_dataset_view(dataset_col->view(), dim);

  std::vector<float> q_host{1.0f, 0.0f};
  rmm::device_uvector<float> q_dev(dim, stream);
  cudaMemcpyAsync(
    q_dev.data(), q_host.data(), sizeof(float) * dim, cudaMemcpyHostToDevice, stream.get());
  auto query_view =
    raft::make_device_matrix_view<const float, int64_t, raft::row_major>(q_dev.data(), 1, dim);

  auto knn = brute_force_knn(res, dataset_view, query_view, /*k=*/3, Metric::CosineExpanded);
  stream.sync();

  auto neighbors = to_host<int64_t>(knn.neighbors->view());
  auto distances = to_host<float>(knn.distances->view());

  // The off-axis row (id 1) is strictly last; the two aligned rows come first.
  REQUIRE(neighbors[2] == 1);
  REQUIRE(distances[0] == Approx(0.0f).margin(1e-4));
  REQUIRE(distances[1] == Approx(0.0f).margin(1e-4));
  REQUIRE(distances[2] > distances[1]);
}

TEST_CASE("brute_force_knn rejects out-of-range k and mismatched dims", "[vss]")
{
  ::cuda::stream_ref stream = cudf::get_default_stream();
  raft::device_resources res{stream};

  constexpr cudf::size_type n_rows = 4;
  constexpr cudf::size_type dim    = 3;
  std::vector<float> data(n_rows * dim, 1.0f);
  auto dataset_col  = make_fixed_size_float_list(data, n_rows, dim);
  auto dataset_view = list_column_as_dataset_view(dataset_col->view(), dim);

  std::vector<float> q_host(dim, 0.0f);
  rmm::device_uvector<float> q_dev(dim, stream);
  cudaMemcpyAsync(
    q_dev.data(), q_host.data(), sizeof(float) * dim, cudaMemcpyHostToDevice, stream.get());
  auto query_view =
    raft::make_device_matrix_view<const float, int64_t, raft::row_major>(q_dev.data(), 1, dim);

  SECTION("k < 1 throws") { REQUIRE_THROWS(brute_force_knn(res, dataset_view, query_view, 0)); }

  SECTION("k > n_rows throws")
  {
    REQUIRE_THROWS(brute_force_knn(res, dataset_view, query_view, n_rows + 1));
  }

  SECTION("query/dataset dimensionality mismatch throws")
  {
    std::vector<float> q2_host(dim + 1, 0.0f);
    rmm::device_uvector<float> q2_dev(dim + 1, stream);
    cudaMemcpyAsync(q2_dev.data(),
                    q2_host.data(),
                    sizeof(float) * (dim + 1),
                    cudaMemcpyHostToDevice,
                    stream.get());
    auto bad_query = raft::make_device_matrix_view<const float, int64_t, raft::row_major>(
      q2_dev.data(), 1, dim + 1);
    REQUIRE_THROWS(brute_force_knn(res, dataset_view, bad_query, 1));
  }
}

TEST_CASE("brute_force_f16_topk returns the exact L2 neighbors of half rows", "[vss][f16]")
{
  // Small integers are exact in half, and so are their norms and distances in FP32. Covers k <= 64
  // (cuVS's fusedL2Knn before NVIDIA/cuvs#3041), k > 64, and k == n.
  auto const shape          = GENERATE(std::array<int64_t, 4>{1, 1, 4, 1},
                              std::array<int64_t, 4>{300, 33, 96, 10},
                              std::array<int64_t, 4>{300, 5, 96, 100},
                              std::array<int64_t, 4>{70, 3, 8, 70});
  auto const [n, m, d, k]   = shape;
  auto const take_sqrt      = GENERATE(false, true);
  ::cuda::stream_ref stream = cudf::get_default_stream();
  raft::device_resources res{stream};

  std::vector<float> x(n * d), q(m * d), x_sq(n, 0.f);
  for (int64_t i = 0; i < n; ++i) {
    for (int64_t j = 0; j < d; ++j) {
      x[i * d + j] = static_cast<float>((i * 13 + j * 19) % 17 - 8);
      x_sq[i] += x[i * d + j] * x[i * d + j];
    }
  }
  for (int64_t i = 0; i < m; ++i) {
    for (int64_t j = 0; j < d; ++j) {
      q[i * d + j] = static_cast<float>((i + j * 7) % 17 - 8);
    }
  }
  auto const to_half_bits = [](std::vector<float> const& v) {
    std::vector<std::uint16_t> bits(v.size());
    for (std::size_t i = 0; i < v.size(); ++i) {
      auto const h = __float2half(v[i]);
      std::memcpy(&bits[i], &h, sizeof(bits[i]));
    }
    return bits;
  };
  auto const x_bits = to_half_bits(x);
  auto const q_bits = to_half_bits(q);
  rmm::device_uvector<std::uint16_t> x_dev(x_bits.size(), stream);
  rmm::device_uvector<std::uint16_t> q_dev(q_bits.size(), stream);
  rmm::device_uvector<float> x_sq_dev(x_sq.size(), stream);
  cudaMemcpyAsync(x_dev.data(),
                  x_bits.data(),
                  sizeof(std::uint16_t) * x_bits.size(),
                  cudaMemcpyHostToDevice,
                  stream.get());
  cudaMemcpyAsync(q_dev.data(),
                  q_bits.data(),
                  sizeof(std::uint16_t) * q_bits.size(),
                  cudaMemcpyHostToDevice,
                  stream.get());
  cudaMemcpyAsync(x_sq_dev.data(),
                  x_sq.data(),
                  sizeof(float) * x_sq.size(),
                  cudaMemcpyHostToDevice,
                  stream.get());

  auto result = sirius::vss::brute_force_f16_topk(
    res, x_dev.data(), x_sq_dev.data(), n, q_dev.data(), m, d, k, take_sqrt);
  stream.sync();
  REQUIRE(result.n_queries == m);
  REQUIRE(result.k == k);
  auto const neighbors = to_host<int64_t>(result.neighbors->view());
  auto const distances = to_host<float>(result.distances->view());
  REQUIRE(neighbors.size() == m * k);
  REQUIRE(distances.size() == m * k);
  for (int64_t row = 0; row < m; ++row) {
    std::vector<float> expected(n);
    for (int64_t col = 0; col < n; ++col) {
      double squared = 0;
      for (int64_t j = 0; j < d; ++j) {
        auto const delta = static_cast<double>(q[row * d + j]) - x[col * d + j];
        squared += delta * delta;
      }
      expected[col] = static_cast<float>(take_sqrt ? std::sqrt(squared) : squared);
    }
    auto sorted = expected;
    std::sort(sorted.begin(), sorted.end());
    std::vector<bool> seen(n, false);
    for (int64_t col = 0; col < k; ++col) {
      auto const id = neighbors[row * k + col];
      REQUIRE(id >= 0);
      REQUIRE(id < n);
      REQUIRE_FALSE(seen[id]);
      seen[id] = true;
      REQUIRE(distances[row * k + col] == Approx(expected[id]));
      REQUIRE(distances[row * k + col] == Approx(sorted[col]));
    }
  }
}
