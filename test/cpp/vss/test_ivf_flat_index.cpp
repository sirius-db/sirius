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
#include <vss/ivf_flat_index.hpp>

// cudf
#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

// rmm
#include <rmm/device_buffer.hpp>

#include <cuda_runtime_api.h>

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace {

// Build a Sirius-style ARRAY<FLOAT>[dim] column (cudf LIST with a contiguous,
// uniform FLOAT32 values child); i.e., the shape build_ivf_flat_index expects.
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

// Upload a host query vector to the device (search wants a device pointer).
rmm::device_buffer upload(std::vector<float> const& v, rmm::cuda_stream_view stream)
{
  rmm::device_buffer buf(v.size() * sizeof(float), stream);
  cudaMemcpyAsync(buf.data(), v.data(), buf.size(), cudaMemcpyHostToDevice, stream.value());
  stream.synchronize();
  return buf;
}

std::vector<int64_t> to_host_i64(cudf::column_view const& col)
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

// The metric string is the CREATE INDEX surface; these mappings must stay in
// lockstep with the recognizer (vss_pattern.cpp::metric_for_function) or a pinned
// index would never match a query's derived metric during auto-routing.
TEST_CASE("ann_distance_type_from_metric maps the supported metrics", "[vss]")
{
  REQUIRE(sirius::vss::ann_distance_type_from_metric("l2sq") ==
          cuvs::distance::DistanceType::L2SqrtExpanded);
  REQUIRE(sirius::vss::ann_distance_type_from_metric("cosine") ==
          cuvs::distance::DistanceType::CosineExpanded);
  REQUIRE_THROWS_AS(sirius::vss::ann_distance_type_from_metric("dot"), std::invalid_argument);
  REQUIRE_THROWS_AS(sirius::vss::ann_distance_type_from_metric(""), std::invalid_argument);
}

TEST_CASE("build_ivf_flat_index + search finds exact nearest neighbours", "[vss]")
{
  constexpr cudf::size_type dim = 2;
  auto const stream             = cudf::get_default_stream();
  auto const mr                 = cudf::get_current_device_resource_ref();

  // Two well-separated clusters: rows 0-2 near the origin, rows 3-5 near (10,10).
  std::vector<float> dataset_vals = {
    0.0f,
    0.0f,  // row 0
    1.0f,
    0.0f,  // row 1
    0.0f,
    1.0f,  // row 2
    10.0f,
    10.0f,  // row 3
    11.0f,
    10.0f,  // row 4
    10.0f,
    11.0f,  // row 5
  };
  auto dataset_col = make_fixed_size_float_list(dataset_vals, 6, dim);

  // n_probes == n_lists probes every list, so the search considers all points and
  // is exact; the ANN approximation only kicks in when fewer lists are probed.
  constexpr std::uint32_t n_lists  = 2;
  constexpr std::uint32_t n_probes = n_lists;
  auto index                       = sirius::vss::build_ivf_flat_index(
    dataset_col->view(), dim, n_lists, cuvs::distance::DistanceType::L2SqrtExpanded, mr);

  SECTION("query in the origin cluster returns row 0")
  {
    auto q      = upload({0.1f, 0.1f}, stream);
    auto result = sirius::vss::search_ivf_flat_index(
      *index, static_cast<const float*>(q.data()), dim, /*k=*/1, n_probes, stream, mr);

    REQUIRE(result.neighbors->size() == 1);
    REQUIRE(result.distances->size() == 1);
    REQUIRE(to_host_i64(result.neighbors->view())[0] == 0);
  }

  SECTION("query in the far cluster returns row 3")
  {
    auto q      = upload({10.1f, 10.1f}, stream);
    auto result = sirius::vss::search_ivf_flat_index(
      *index, static_cast<const float*>(q.data()), dim, /*k=*/1, n_probes, stream, mr);

    REQUIRE(to_host_i64(result.neighbors->view())[0] == 3);
  }

  SECTION("k = 3 for an origin query stays within the origin cluster, ordered")
  {
    auto q      = upload({0.1f, 0.1f}, stream);
    auto result = sirius::vss::search_ivf_flat_index(
      *index, static_cast<const float*>(q.data()), dim, /*k=*/3, n_probes, stream, mr);

    auto neighbors = to_host_i64(result.neighbors->view());
    auto distances = to_host_f(result.distances->view());
    REQUIRE(neighbors[0] == 0);
    REQUIRE(neighbors[1] < 3);
    REQUIRE(neighbors[2] < 3);
    REQUIRE(distances[0] <= distances[1]);
    REQUIRE(distances[1] <= distances[2]);
  }
}

// Pin the distance magnitude: L2SqrtExpanded must yield Euclidean (rooted)
// distances, which is the array_distance contract the ANN operator relies on.
// Ordering-only checks would let a squared-L2 regression pass silently.
TEST_CASE("search_ivf_flat_index returns Euclidean distances for L2SqrtExpanded", "[vss]")
{
  constexpr cudf::size_type dim = 2;
  auto const stream             = cudf::get_default_stream();
  auto const mr                 = cudf::get_current_device_resource_ref();

  // Pythagorean rows so the distances to the origin are exact: 0, 5, 10, 13.
  std::vector<float> dataset_vals = {
    0.0f,
    0.0f,  // row 0 -> 0
    3.0f,
    4.0f,  // row 1 -> 5
    8.0f,
    6.0f,  // row 2 -> 10
    5.0f,
    12.0f,  // row 3 -> 13
  };
  auto dataset_col = make_fixed_size_float_list(dataset_vals, 4, dim);

  // Single list so every point is in it; probing it is a full (exact) scan.
  auto index = sirius::vss::build_ivf_flat_index(
    dataset_col->view(), dim, /*n_lists=*/1, cuvs::distance::DistanceType::L2SqrtExpanded, mr);

  auto q      = upload({0.0f, 0.0f}, stream);
  auto result = sirius::vss::search_ivf_flat_index(
    *index, static_cast<const float*>(q.data()), dim, /*k=*/4, /*n_probes=*/1, stream, mr);

  auto neighbors = to_host_i64(result.neighbors->view());
  auto distances = to_host_f(result.distances->view());
  REQUIRE(neighbors == std::vector<int64_t>{0, 1, 2, 3});
  REQUIRE(distances[0] == Approx(0.0f));
  REQUIRE(distances[1] == Approx(5.0f));
  REQUIRE(distances[2] == Approx(10.0f));
  REQUIRE(distances[3] == Approx(13.0f));
}

TEST_CASE("search_ivf_flat_index handles k == n_rows", "[vss]")
{
  constexpr cudf::size_type dim = 1;
  auto const stream             = cudf::get_default_stream();
  auto const mr                 = cudf::get_current_device_resource_ref();

  std::vector<float> dataset_vals = {1.0f, 5.0f, 2.0f};  // distances to 0: 1, 5, 2
  auto dataset_col                = make_fixed_size_float_list(dataset_vals, 3, dim);
  auto index                      = sirius::vss::build_ivf_flat_index(
    dataset_col->view(), dim, /*n_lists=*/1, cuvs::distance::DistanceType::L2SqrtExpanded, mr);

  auto q      = upload({0.0f}, stream);
  auto result = sirius::vss::search_ivf_flat_index(
    *index, static_cast<const float*>(q.data()), dim, /*k=*/3, /*n_probes=*/1, stream, mr);

  auto neighbors = to_host_i64(result.neighbors->view());
  auto distances = to_host_f(result.distances->view());
  REQUIRE(neighbors == std::vector<int64_t>{0, 2, 1});  // 1 < 2 < 5
  REQUIRE(distances[0] == Approx(1.0f));
  REQUIRE(distances[1] == Approx(2.0f));
  REQUIRE(distances[2] == Approx(5.0f));
}

// cuVS IVF-Flat supports CosineExpanded; verify it ranks by direction (nearest =
// smallest angle) so the operator's ascending-by-distance sort is correct for
// cosine too. Magnitudes are left unpinned (cuVS owns the cosine-distance scale),
// but the strictly-separated angles make the neighbour ORDER unambiguous.
TEST_CASE("build/search IVF-Flat ranks by direction for CosineExpanded", "[vss]")
{
  constexpr cudf::size_type dim = 2;
  auto const stream             = cudf::get_default_stream();
  auto const mr                 = cudf::get_current_device_resource_ref();

  // Directions at 0, 45, 90, 135 degrees from the query [1,0].
  std::vector<float> dataset_vals = {
    1.0f,
    0.0f,  // row 0 -> 0 deg (identical direction)
    1.0f,
    1.0f,  // row 1 -> 45 deg
    0.0f,
    1.0f,  // row 2 -> 90 deg
    -1.0f,
    1.0f,  // row 3 -> 135 deg
  };
  auto dataset_col = make_fixed_size_float_list(dataset_vals, 4, dim);
  auto index       = sirius::vss::build_ivf_flat_index(
    dataset_col->view(), dim, /*n_lists=*/1, cuvs::distance::DistanceType::CosineExpanded, mr);

  auto q      = upload({1.0f, 0.0f}, stream);
  auto result = sirius::vss::search_ivf_flat_index(
    *index, static_cast<const float*>(q.data()), dim, /*k=*/4, /*n_probes=*/1, stream, mr);

  auto neighbors = to_host_i64(result.neighbors->view());
  auto distances = to_host_f(result.distances->view());
  REQUIRE(neighbors == std::vector<int64_t>{0, 1, 2, 3});  // increasing angle
  REQUIRE(distances[0] <= distances[1]);
  REQUIRE(distances[1] <= distances[2]);
  REQUIRE(distances[2] <= distances[3]);
}

TEST_CASE("build_ivf_flat_index rejects a dim that mismatches the column width", "[vss]")
{
  constexpr cudf::size_type dim = 2;
  auto const mr                 = cudf::get_current_device_resource_ref();
  std::vector<float> vals       = {0.0f, 0.0f, 1.0f, 1.0f};
  auto col                      = make_fixed_size_float_list(vals, 2, dim);

  // Column rows are width 2; asking the builder to read them as width 3 must be
  // rejected by the dataset-view validation, not silently misinterpreted.
  REQUIRE_THROWS(sirius::vss::build_ivf_flat_index(
    col->view(), /*dim=*/3, /*n_lists=*/1, cuvs::distance::DistanceType::L2SqrtExpanded, mr));
}

TEST_CASE("search_ivf_flat_index rejects an index that is not IVF-Flat", "[vss]")
{
  auto const stream = cudf::get_default_stream();
  auto const mr     = cudf::get_current_device_resource_ref();

  // A type-erased holder of the wrong concrete type: the search's dynamic_cast
  // back to the IVF-Flat index must fail loudly rather than reinterpret memory.
  auto not_ivf_flat = sirius::vss::make_cuvs_index(int{7});
  auto q            = upload({0.0f}, stream);
  REQUIRE_THROWS_AS(sirius::vss::search_ivf_flat_index(*not_ivf_flat,
                                                       static_cast<const float*>(q.data()),
                                                       /*dim=*/1,
                                                       /*k=*/1,
                                                       /*n_probes=*/1,
                                                       stream,
                                                       mr),
                    std::invalid_argument);
}
