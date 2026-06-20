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

// rmm
#include <rmm/device_buffer.hpp>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace {

// Build a Sirius-style ARRAY<FLOAT>[dim] column (cudf LIST with a contiguous,
// uniform FLOAT32 values child).
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

std::vector<int64_t> to_host(cudf::column_view const& col)
{
  std::vector<int64_t> host(col.size());
  cudaMemcpy(
    host.data(), col.data<int64_t>(), sizeof(int64_t) * host.size(), cudaMemcpyDeviceToHost);
  return host;
}

std::vector<float> to_host_f(cudf::column_view const& col)
{
  std::vector<float> host(col.size());
  cudaMemcpy(host.data(), col.data<float>(), sizeof(float) * host.size(), cudaMemcpyDeviceToHost);
  return host;
}

}  // namespace

TEST_CASE("brute_force_knn finds exact nearest neighbours", "[vss]")
{
  constexpr cudf::size_type dim = 2;

  // Two well-separated clusters: rows 0-2 near the origin, rows 3-4 near (10,10).
  std::vector<float> dataset_vals = {
    0.0f,
    0.0f,  // row 0
    1.0f,
    0.0f,  // row 1
    0.0f,
    1.0f,  // row 2
    10.0f,
    10.0f,  // row 3
    10.0f,
    11.0f  // row 4
  };
  auto dataset_col = make_fixed_size_float_list(dataset_vals, 5, dim);

  // q0 sits next to row 0; q1 sits next to row 3.
  std::vector<float> query_vals = {
    0.1f,
    0.1f,  // q0
    10.0f,
    10.4f  // q1
  };
  auto query_col = make_fixed_size_float_list(query_vals, 2, dim);

  auto dataset_view = sirius::vss::list_column_as_dataset_view(dataset_col->view(), dim);
  auto query_view   = sirius::vss::list_column_as_dataset_view(query_col->view(), dim);

  SECTION("k = 1 returns the single closest row per query")
  {
    auto result = sirius::vss::brute_force_knn(dataset_view, query_view, /*k=*/1);

    REQUIRE(result.n_queries == 2);
    REQUIRE(result.k == 1);
    REQUIRE(result.neighbors->size() == 2);
    REQUIRE(result.distances->size() == 2);

    auto neighbors = to_host(result.neighbors->view());
    REQUIRE(neighbors[0] == 0);  // q0 -> row 0
    REQUIRE(neighbors[1] == 3);  // q1 -> row 3
  }

  SECTION("k = 3 returns the nearest cluster first, ordered by distance")
  {
    auto result = sirius::vss::brute_force_knn(dataset_view, query_view, /*k=*/3);

    REQUIRE(result.neighbors->size() == 6);
    auto neighbors = to_host(result.neighbors->view());
    auto distances = to_host_f(result.distances->view());

    // q0's nearest is row 0, and its 3 neighbours are all from the origin
    // cluster {0,1,2}.
    REQUIRE(neighbors[0] == 0);
    REQUIRE(neighbors[1] < 3);
    REQUIRE(neighbors[2] < 3);

    // Distances are non-decreasing within each query block.
    REQUIRE(distances[0] <= distances[1]);
    REQUIRE(distances[1] <= distances[2]);
  }
}

// The metric decides whether returned distances are squared or rooted. The SQL
// surface relies on this exact contract: `array_distance` is wired to
// L2SqrtExpanded (Euclidean), `array_cosine_distance` to CosineExpanded. Without
// magnitude assertions, a metric/sqrt regression (e.g. defaulting to the squared
// L2Expanded) would silently still pass ordering-only checks, so pin the values.
TEST_CASE("brute_force_knn distance magnitudes match the requested metric", "[vss]")
{
  constexpr cudf::size_type dim = 2;

  // Pythagorean rows so both squared and rooted distances are exact in float:
  // distances to the origin are 0, 5, 10, 13.
  std::vector<float> dataset_vals = {
    0.0f,
    0.0f,  // row 0 -> 0
    3.0f,
    4.0f,  // row 1 -> 5  (25)
    8.0f,
    6.0f,  // row 2 -> 10 (100)
    5.0f,
    12.0f,  // row 3 -> 13 (169)
  };
  auto dataset_col  = make_fixed_size_float_list(dataset_vals, 4, dim);
  auto dataset_view = sirius::vss::list_column_as_dataset_view(dataset_col->view(), dim);

  std::vector<float> query_vals = {0.0f, 0.0f};
  auto query_col                = make_fixed_size_float_list(query_vals, 1, dim);
  auto query_view               = sirius::vss::list_column_as_dataset_view(query_col->view(), dim);

  SECTION("L2SqrtExpanded returns Euclidean distance (the array_distance contract)")
  {
    auto result = sirius::vss::brute_force_knn(
      dataset_view, query_view, /*k=*/4, cuvs::distance::DistanceType::L2SqrtExpanded);

    auto neighbors = to_host(result.neighbors->view());
    auto distances = to_host_f(result.distances->view());
    REQUIRE(neighbors == std::vector<int64_t>{0, 1, 2, 3});
    REQUIRE(distances[0] == Approx(0.0f));
    REQUIRE(distances[1] == Approx(5.0f));
    REQUIRE(distances[2] == Approx(10.0f));
    REQUIRE(distances[3] == Approx(13.0f));
  }

  SECTION("L2Expanded returns squared L2 distance")
  {
    auto result = sirius::vss::brute_force_knn(
      dataset_view, query_view, /*k=*/4, cuvs::distance::DistanceType::L2Expanded);

    auto distances = to_host_f(result.distances->view());
    REQUIRE(distances[0] == Approx(0.0f));
    REQUIRE(distances[1] == Approx(25.0f));
    REQUIRE(distances[2] == Approx(100.0f));
    REQUIRE(distances[3] == Approx(169.0f));
  }
}

TEST_CASE("brute_force_knn handles boundary shapes", "[vss]")
{
  SECTION("k == n_rows returns every row, ordered by distance")
  {
    constexpr cudf::size_type dim   = 1;
    std::vector<float> dataset_vals = {1.0f, 5.0f, 2.0f};  // distances to 0: 1, 5, 2
    auto dataset_col                = make_fixed_size_float_list(dataset_vals, 3, dim);
    auto dataset_view = sirius::vss::list_column_as_dataset_view(dataset_col->view(), dim);

    std::vector<float> query_vals = {0.0f};
    auto query_col                = make_fixed_size_float_list(query_vals, 1, dim);
    auto query_view = sirius::vss::list_column_as_dataset_view(query_col->view(), dim);

    auto result = sirius::vss::brute_force_knn(
      dataset_view, query_view, /*k=*/3, cuvs::distance::DistanceType::L2SqrtExpanded);

    auto neighbors = to_host(result.neighbors->view());
    auto distances = to_host_f(result.distances->view());
    REQUIRE(neighbors == std::vector<int64_t>{0, 2, 1});  // 1 < 2 < 5
    REQUIRE(distances[0] == Approx(1.0f));
    REQUIRE(distances[1] == Approx(2.0f));
    REQUIRE(distances[2] == Approx(5.0f));
  }

  SECTION("single-row dataset with k = 1")
  {
    constexpr cudf::size_type dim   = 2;
    std::vector<float> dataset_vals = {7.0f, 7.0f};
    auto dataset_col                = make_fixed_size_float_list(dataset_vals, 1, dim);
    auto dataset_view = sirius::vss::list_column_as_dataset_view(dataset_col->view(), dim);

    std::vector<float> query_vals = {0.0f, 0.0f};
    auto query_col                = make_fixed_size_float_list(query_vals, 1, dim);
    auto query_view = sirius::vss::list_column_as_dataset_view(query_col->view(), dim);

    auto result = sirius::vss::brute_force_knn(
      dataset_view, query_view, /*k=*/1, cuvs::distance::DistanceType::L2SqrtExpanded);

    REQUIRE(result.neighbors->size() == 1);
    auto neighbors = to_host(result.neighbors->view());
    auto distances = to_host_f(result.distances->view());
    REQUIRE(neighbors[0] == 0);
    REQUIRE(distances[0] == Approx(std::sqrt(98.0f)));
  }
}

TEST_CASE("brute_force_knn breaks distance ties without dropping rows", "[vss]")
{
  constexpr cudf::size_type dim = 2;
  // Rows 0 and 1 are equidistant from the origin (both distance 1); the metric
  // must surface both, not collapse the tie.
  std::vector<float> dataset_vals = {1.0f, 0.0f, 0.0f, 1.0f};
  auto dataset_col                = make_fixed_size_float_list(dataset_vals, 2, dim);
  auto dataset_view = sirius::vss::list_column_as_dataset_view(dataset_col->view(), dim);

  std::vector<float> query_vals = {0.0f, 0.0f};
  auto query_col                = make_fixed_size_float_list(query_vals, 1, dim);
  auto query_view               = sirius::vss::list_column_as_dataset_view(query_col->view(), dim);

  auto result = sirius::vss::brute_force_knn(
    dataset_view, query_view, /*k=*/2, cuvs::distance::DistanceType::L2SqrtExpanded);

  auto neighbors = to_host(result.neighbors->view());
  auto distances = to_host_f(result.distances->view());

  std::sort(neighbors.begin(), neighbors.end());  // tie order is unspecified
  REQUIRE(neighbors == std::vector<int64_t>{0, 1});
  REQUIRE(distances[0] == Approx(1.0f));
  REQUIRE(distances[1] == Approx(1.0f));
}

TEST_CASE("brute_force_knn rejects invalid k", "[vss]")
{
  constexpr cudf::size_type dim   = 2;
  std::vector<float> dataset_vals = {0.0f, 0.0f, 1.0f, 1.0f};
  auto dataset_col                = make_fixed_size_float_list(dataset_vals, 2, dim);
  auto dataset_view = sirius::vss::list_column_as_dataset_view(dataset_col->view(), dim);

  REQUIRE_THROWS(sirius::vss::brute_force_knn(dataset_view, dataset_view, /*k=*/0));
  REQUIRE_THROWS(sirius::vss::brute_force_knn(dataset_view, dataset_view, /*k=*/3));
}
