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
#include <vss/brute_force_threshold.hpp>
#include <vss/cudf_raft_interop.hpp>

// cudf
#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

// raft / rmm / cuvs
#include <raft/core/device_resources.hpp>

#include <rmm/device_buffer.hpp>

#include <cuda_runtime_api.h>

#include <cuvs/distance/distance.hpp>

#include <cmath>
#include <cstdint>
#include <map>
#include <utility>
#include <vector>

namespace {

using sirius::vss::brute_force_threshold;
using sirius::vss::list_column_as_dataset_view;
using sirius::vss::threshold_join_result;
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

template <typename T>
std::vector<T> to_host(cudf::column_view const& col)
{
  std::vector<T> out(col.size());
  if (!out.empty()) {
    cudaMemcpy(out.data(), col.data<T>(), sizeof(T) * out.size(), cudaMemcpyDeviceToHost);
  }
  return out;
}

// The result is an unordered ragged edge list; fold it into query -> sorted
// (neighbor, distance) so tests can assert on it deterministically.
using edge_map = std::map<int64_t, std::map<int64_t, float>>;

edge_map collect_edges(threshold_join_result const& r)
{
  auto const q = to_host<int64_t>(r.query_rows->view());
  auto const n = to_host<int64_t>(r.neighbors->view());
  auto const d = to_host<float>(r.distances->view());
  REQUIRE(q.size() == static_cast<std::size_t>(r.n_edges));
  REQUIRE(n.size() == q.size());
  REQUIRE(d.size() == q.size());

  edge_map edges;
  for (std::size_t i = 0; i < q.size(); ++i) {
    edges[q[i]][n[i]] = d[i];
  }
  return edges;
}

}  // namespace

TEST_CASE("brute_force_threshold keeps every L2 neighbor within the radius", "[vss]")
{
  auto stream = cudf::get_default_stream();
  raft::device_resources res{stream};
  auto const mr = cudf::get_current_device_resource_ref();

  // Dataset row i is [i, i, i]; the single query is the origin, so the distance
  // to row i is i * sqrt(3). eps = 4.0 keeps rows 0, 1, 2 (0, 1.73, 3.46) and
  // drops row 3 (5.196) onward.
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

  std::vector<float> const q{0.0f, 0.0f, 0.0f};
  auto query_col  = make_fixed_size_float_list(q, 1, dim);
  auto query_view = list_column_as_dataset_view(query_col->view(), dim);

  // Same expected result whether L2 is computed unexpanded (no GEMM) or expanded (GEMM).
  auto metric = Metric::L2SqrtUnexpanded;
  SECTION("L2 unexpanded (no GEMM)") { metric = Metric::L2SqrtUnexpanded; }
  SECTION("L2 expanded (GEMM)") { metric = Metric::L2SqrtExpanded; }

  auto r = brute_force_threshold(res, dataset_view, query_view, /*eps=*/4.0f, metric, mr);
  stream.synchronize();

  auto const edges = collect_edges(r);
  REQUIRE(r.n_edges == 3);
  REQUIRE(edges.size() == 1);
  auto const& q0 = edges.at(0);
  REQUIRE(q0.size() == 3);
  REQUIRE(q0.count(0) == 1);
  REQUIRE(q0.count(1) == 1);
  REQUIRE(q0.count(2) == 1);
  REQUIRE(q0.at(0) == Approx(0.0f).margin(1e-3));
  REQUIRE(q0.at(1) == Approx(std::sqrt(3.0f)).margin(1e-3));
  REQUIRE(q0.at(2) == Approx(2.0f * std::sqrt(3.0f)).margin(1e-3));
}

TEST_CASE("brute_force_threshold cosine radius ignores magnitude", "[vss]")
{
  auto stream = cudf::get_default_stream();
  raft::device_resources res{stream};
  auto const mr = cudf::get_current_device_resource_ref();

  // Row 0 aligned with the query, row 1 at 45 deg, row 2 aligned but large.
  // Cosine distance: rows 0 and 2 are 0 (magnitude ignored), row 1 is ~0.293.
  // A 0.1 cutoff keeps the two aligned rows and drops the off-axis one.
  constexpr cudf::size_type dim    = 2;
  constexpr cudf::size_type n_rows = 3;
  std::vector<float> const data{1.0f, 0.0f, 1.0f, 1.0f, 9.0f, 0.0f};
  auto dataset_col  = make_fixed_size_float_list(data, n_rows, dim);
  auto dataset_view = list_column_as_dataset_view(dataset_col->view(), dim);

  std::vector<float> const q{1.0f, 0.0f};
  auto query_col  = make_fixed_size_float_list(q, 1, dim);
  auto query_view = list_column_as_dataset_view(query_col->view(), dim);

  auto r =
    brute_force_threshold(res, dataset_view, query_view, /*eps=*/0.1f, Metric::CosineExpanded, mr);
  stream.synchronize();

  auto const edges = collect_edges(r);
  REQUIRE(r.n_edges == 2);
  auto const& q0 = edges.at(0);
  REQUIRE(q0.count(0) == 1);
  REQUIRE(q0.count(2) == 1);
  REQUIRE(q0.count(1) == 0);
  REQUIRE(q0.at(0) == Approx(0.0f).margin(1e-4));
  REQUIRE(q0.at(2) == Approx(0.0f).margin(1e-4));
}

TEST_CASE("brute_force_threshold emits a ragged, per-query-variable edge list", "[vss]")
{
  auto stream = cudf::get_default_stream();
  raft::device_resources res{stream};
  auto const mr = cudf::get_current_device_resource_ref();

  // 1-D points 0..5. Query 1.0 has neighbors {0,1,2} within 1.5; query 5.0 has
  // {4,5}. Different counts per query exercises the ragged output.
  constexpr cudf::size_type dim    = 1;
  constexpr cudf::size_type n_rows = 6;
  std::vector<float> const data{0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
  auto dataset_col  = make_fixed_size_float_list(data, n_rows, dim);
  auto dataset_view = list_column_as_dataset_view(dataset_col->view(), dim);

  std::vector<float> const queries{1.0f, 5.0f};
  auto query_col  = make_fixed_size_float_list(queries, 2, dim);
  auto query_view = list_column_as_dataset_view(query_col->view(), dim);

  // Force tiny tiles too, so the same result must come out of the tiled append path.
  std::size_t tile_rows = 0;
  std::size_t tile_cols = 0;
  SECTION("auto tile size") {}
  SECTION("forced small tiles (1x2)")
  {
    tile_rows = 1;
    tile_cols = 2;
  }

  auto r = brute_force_threshold(res,
                                 dataset_view,
                                 query_view,
                                 /*eps=*/1.5f,
                                 Metric::L2SqrtExpanded,
                                 mr,
                                 tile_rows,
                                 tile_cols);
  stream.synchronize();

  auto const edges = collect_edges(r);
  REQUIRE(r.n_edges == 5);
  REQUIRE(edges.at(0).size() == 3);  // query 1.0 -> {0,1,2}
  REQUIRE(edges.at(0).count(0) == 1);
  REQUIRE(edges.at(0).count(1) == 1);
  REQUIRE(edges.at(0).count(2) == 1);
  REQUIRE(edges.at(1).size() == 2);  // query 5.0 -> {4,5}
  REQUIRE(edges.at(1).count(4) == 1);
  REQUIRE(edges.at(1).count(5) == 1);
}

TEST_CASE("brute_force_threshold returns no edges when nothing is within the radius", "[vss]")
{
  auto stream = cudf::get_default_stream();
  raft::device_resources res{stream};
  auto const mr = cudf::get_current_device_resource_ref();

  constexpr cudf::size_type dim    = 1;
  constexpr cudf::size_type n_rows = 3;
  std::vector<float> const data{0.0f, 1.0f, 2.0f};
  auto dataset_col  = make_fixed_size_float_list(data, n_rows, dim);
  auto dataset_view = list_column_as_dataset_view(dataset_col->view(), dim);

  std::vector<float> const q{100.0f};
  auto query_col  = make_fixed_size_float_list(q, 1, dim);
  auto query_view = list_column_as_dataset_view(query_col->view(), dim);

  auto r =
    brute_force_threshold(res, dataset_view, query_view, /*eps=*/1.0f, Metric::L2SqrtExpanded, mr);
  stream.synchronize();

  REQUIRE(r.n_edges == 0);
  REQUIRE(r.query_rows->size() == 0);
  REQUIRE(r.neighbors->size() == 0);
  REQUIRE(r.distances->size() == 0);
}
