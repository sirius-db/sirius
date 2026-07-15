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
#include <vss/brute_force_search.hpp>
#include <vss/cudf_raft_interop.hpp>

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

#include <cuda_runtime_api.h>

#include <cuvs/distance/distance.hpp>

#include <cmath>
#include <cstdint>
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
  auto stream = cudf::get_default_stream();
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
    q_dev.data(), q_host.data(), sizeof(float) * dim, cudaMemcpyHostToDevice, stream.value());
  auto query_view =
    raft::make_device_matrix_view<const float, int64_t, raft::row_major>(q_dev.data(), 1, dim);

  constexpr int64_t k = 3;
  auto knn            = brute_force_knn(res, dataset_view, query_view, k, Metric::L2SqrtUnexpanded);
  stream.synchronize();

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
  auto stream = cudf::get_default_stream();
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
    q_dev.data(), q_host.data(), sizeof(float) * dim, cudaMemcpyHostToDevice, stream.value());
  auto query_view =
    raft::make_device_matrix_view<const float, int64_t, raft::row_major>(q_dev.data(), 1, dim);

  auto knn = brute_force_knn(res, dataset_view, query_view, /*k=*/3, Metric::CosineExpanded);
  stream.synchronize();

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
  auto stream = cudf::get_default_stream();
  raft::device_resources res{stream};

  constexpr cudf::size_type n_rows = 4;
  constexpr cudf::size_type dim    = 3;
  std::vector<float> data(n_rows * dim, 1.0f);
  auto dataset_col  = make_fixed_size_float_list(data, n_rows, dim);
  auto dataset_view = list_column_as_dataset_view(dataset_col->view(), dim);

  std::vector<float> q_host(dim, 0.0f);
  rmm::device_uvector<float> q_dev(dim, stream);
  cudaMemcpyAsync(
    q_dev.data(), q_host.data(), sizeof(float) * dim, cudaMemcpyHostToDevice, stream.value());
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
                    stream.value());
    auto bad_query = raft::make_device_matrix_view<const float, int64_t, raft::row_major>(
      q2_dev.data(), 1, dim + 1);
    REQUIRE_THROWS(brute_force_knn(res, dataset_view, bad_query, 1));
  }
}
