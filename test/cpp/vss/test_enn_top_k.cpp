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
#include "operator/operator_test_utils.hpp"

#include <catch.hpp>

// sirius
#include <scan_manager/sirius_scan_manager.hpp>
#include <sirius_context.hpp>
#include <vss/enn_top_k.hpp>
#include <vss/vector_search.hpp>
#include <vss/vector_search_internal.hpp>

// cudf
#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

// raft / rmm
#include <raft/core/device_resources.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>

#include <cuda_runtime_api.h>

#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

namespace {

namespace ou = sirius::test::operator_utils;
using sirius::vss::compute_enn_top_k;
using sirius::vss::merge_enn_top_k;
using sirius::vss::vector_search_context;
using sirius::vss::vector_search_request;

// Sirius-style ARRAY<FLOAT>[dim] column (cudf LIST, contiguous FLOAT32 child).
std::unique_ptr<cudf::column> make_vec_column(std::vector<float> const& values,
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

std::unique_ptr<cudf::column> make_int32_column(std::vector<int32_t> const& values)
{
  auto col = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32},
                                       static_cast<cudf::size_type>(values.size()),
                                       cudf::mask_state::UNALLOCATED);
  cudaMemcpy(col->mutable_view().data<int32_t>(),
             values.data(),
             sizeof(int32_t) * values.size(),
             cudaMemcpyHostToDevice);
  return col;
}

std::unique_ptr<cudf::column> make_float_column(std::vector<float> const& values)
{
  auto col = cudf::make_numeric_column(cudf::data_type{cudf::type_id::FLOAT32},
                                       static_cast<cudf::size_type>(values.size()),
                                       cudf::mask_state::UNALLOCATED);
  cudaMemcpy(col->mutable_view().data<float>(),
             values.data(),
             sizeof(float) * values.size(),
             cudaMemcpyHostToDevice);
  return col;
}

template <typename T>
std::vector<T> to_host(cudf::column_view const& col)
{
  std::vector<T> out(col.size());
  cudaMemcpy(out.data(), col.data<T>(), sizeof(T) * out.size(), cudaMemcpyDeviceToHost);
  return out;
}

// Owns the durable pieces a vector_search_context references (ctx, pin, request,
// uploaded query) so compute_enn_top_k / merge_enn_top_k can be driven directly.
// compute/merge only read stream, mr, k, req.dim, req.metric, query_device, so
// ctx/pin/host_space are inert stand-ins.
struct EnnHarness {
  duckdb::SiriusContext ctx;  // default-constructed, never initialized
  sirius::scan_manager::pinned_entry pin;
  vector_search_request req;
  cucascade::memory::memory_space* space = ou::get_default_gpu_space();
  rmm::device_async_resource_ref mr      = ou::get_resource_ref(*space);
  rmm::cuda_stream_view stream           = ou::default_stream();
  rmm::device_uvector<float> query_dev{0, ou::default_stream()};

  EnnHarness(cudf::size_type dim, std::string metric, std::vector<float> query)
  {
    req.dim    = dim;
    req.metric = std::move(metric);
    query_dev  = rmm::device_uvector<float>(query.size(), stream);
    cudaMemcpyAsync(query_dev.data(),
                    query.data(),
                    sizeof(float) * query.size(),
                    cudaMemcpyHostToDevice,
                    stream.value());
  }

  vector_search_context context(std::int64_t k)
  {
    return vector_search_context{
      ctx, req, *space, *space, pin, mr, stream, query_dev.data(), /*target_gpu=*/0, k};
  }
};

}  // namespace

TEST_CASE("compute_enn_top_k returns nearest passthrough rows plus distance", "[vss]")
{
  constexpr cudf::size_type n_rows = 6;
  constexpr cudf::size_type dim    = 3;
  std::vector<float> vecs(n_rows * dim);
  std::vector<int32_t> ids(n_rows);
  for (cudf::size_type i = 0; i < n_rows; ++i) {
    vecs[i * dim + 0] = static_cast<float>(i);
    vecs[i * dim + 1] = static_cast<float>(i);
    vecs[i * dim + 2] = static_cast<float>(i);
    ids[i]            = i;
  }

  EnnHarness h(dim, "l2", std::vector<float>(dim, 0.0f));
  raft::device_resources res{h.stream};

  SECTION("top-k in nearest-first order with the trailing distance column")
  {
    auto vec_col = make_vec_column(vecs, n_rows, dim);
    auto id_col  = make_int32_column(ids);
    cudf::table_view input({vec_col->view(), id_col->view()});

    auto out = compute_enn_top_k(h.context(/*k=*/3), input, res);
    h.stream.synchronize();

    // Schema is [id, distance] (the vector column is dropped, distance appended).
    REQUIRE(out->num_columns() == 2);
    REQUIRE(out->num_rows() == 3);
    REQUIRE(out->get_column(0).type().id() == cudf::type_id::INT32);
    REQUIRE(out->get_column(1).type().id() == cudf::type_id::FLOAT32);

    auto got_ids  = to_host<int32_t>(out->get_column(0).view());
    auto got_dist = to_host<float>(out->get_column(1).view());
    REQUIRE(got_ids == std::vector<int32_t>{0, 1, 2});
    for (int i = 0; i < 3; ++i) {
      REQUIRE(got_dist[i] == Approx(static_cast<float>(i) * std::sqrt(3.0f)).margin(1e-3));
    }
  }

  SECTION("k larger than the row count clamps to all rows")
  {
    auto vec_col = make_vec_column(vecs, n_rows, dim);
    auto id_col  = make_int32_column(ids);
    cudf::table_view input({vec_col->view(), id_col->view()});

    auto out = compute_enn_top_k(h.context(/*k=*/100), input, res);
    h.stream.synchronize();
    REQUIRE(out->num_rows() == n_rows);
  }

  SECTION("k == 0 yields the empty [id, distance] schema")
  {
    auto vec_col = make_vec_column(vecs, n_rows, dim);
    auto id_col  = make_int32_column(ids);
    cudf::table_view input({vec_col->view(), id_col->view()});

    auto out = compute_enn_top_k(h.context(/*k=*/0), input, res);
    REQUIRE(out->num_rows() == 0);
    REQUIRE(out->num_columns() == 2);
  }
}

TEST_CASE("compute_enn_top_k compacts null and sliced vector rows before search", "[vss]")
{
  constexpr cudf::size_type n_rows = 5;
  constexpr cudf::size_type dim    = 3;
  std::vector<float> vecs(n_rows * dim);
  std::vector<int32_t> ids(n_rows);
  for (cudf::size_type i = 0; i < n_rows; ++i) {
    vecs[i * dim + 0] = static_cast<float>(i);
    vecs[i * dim + 1] = static_cast<float>(i);
    vecs[i * dim + 2] = static_cast<float>(i);
    ids[i]            = i;
  }

  EnnHarness h(dim, "l2", std::vector<float>(dim, 0.0f));
  raft::device_resources res{h.stream};

  SECTION("a null vector row is dropped and never returned")
  {
    // Mark row 2 (id 2) null. Compaction drops it but keeps id alignment, so the
    // three nearest survivors are ids 0, 1, 3.
    auto vec_col  = make_vec_column(vecs, n_rows, dim);
    auto contents = vec_col->release();
    auto mask     = cudf::create_null_mask(n_rows, cudf::mask_state::ALL_VALID);
    cudf::set_null_mask(static_cast<cudf::bitmask_type*>(mask.data()), 2, 3, false);
    auto null_vec = cudf::make_lists_column(
      n_rows, std::move(contents.children[0]), std::move(contents.children[1]), 1, std::move(mask));
    auto id_col = make_int32_column(ids);
    cudf::table_view input({null_vec->view(), id_col->view()});

    auto out = compute_enn_top_k(h.context(/*k=*/3), input, res);
    h.stream.synchronize();

    auto got_ids = to_host<int32_t>(out->get_column(0).view());
    REQUIRE(got_ids == std::vector<int32_t>{0, 1, 3});
  }

  SECTION("a sliced (non-zero offset) input is compacted and stays row-aligned")
  {
    auto vec_col = make_vec_column(vecs, n_rows, dim);
    auto id_col  = make_int32_column(ids);
    // Drop the first row: parent offset becomes 1, forcing the compaction path.
    auto sliced_vec = cudf::slice(vec_col->view(), {1, n_rows}).front();
    auto sliced_id  = cudf::slice(id_col->view(), {1, n_rows}).front();
    REQUIRE(sliced_vec.offset() != 0);
    cudf::table_view input({sliced_vec, sliced_id});

    auto out = compute_enn_top_k(h.context(/*k=*/2), input, res);
    h.stream.synchronize();

    auto got_ids = to_host<int32_t>(out->get_column(0).view());
    REQUIRE(got_ids == std::vector<int32_t>{1, 2});
  }
}

TEST_CASE("merge_enn_top_k keeps the globally nearest rows sorted by distance", "[vss]")
{
  EnnHarness h(/*dim=*/3, "l2", std::vector<float>(3, 0.0f));

  // Two per-chunk candidate tables [id, distance], each already sorted ascending
  // by distance (as cuVS select_k returns them). Global order across both chunks:
  // id 3 (0.0), id 1 (1.0), id 2 (2.0), id 0 (3.0).
  std::vector<int32_t> ids_a{3, 2};
  std::vector<float> dist_a{0.0f, 2.0f};
  std::vector<int32_t> ids_b{1, 0};
  std::vector<float> dist_b{1.0f, 3.0f};

  auto id_a = make_int32_column(ids_a);
  auto d_a  = make_float_column(dist_a);
  auto id_b = make_int32_column(ids_b);
  auto d_b  = make_float_column(dist_b);
  std::vector<cudf::table_view> candidates{cudf::table_view({id_a->view(), d_a->view()}),
                                           cudf::table_view({id_b->view(), d_b->view()})};

  SECTION("top-k by ascending distance, sorted nearest-first")
  {
    auto out = merge_enn_top_k(h.context(/*k=*/2), candidates);
    h.stream.synchronize();

    REQUIRE(out->num_rows() == 2);
    auto got_ids  = to_host<int32_t>(out->get_column(0).view());
    auto got_dist = to_host<float>(out->get_column(1).view());
    REQUIRE(got_ids == std::vector<int32_t>{3, 1});
    REQUIRE(got_dist[0] == Approx(0.0f).margin(1e-4));
    REQUIRE(got_dist[1] == Approx(1.0f).margin(1e-4));
  }

  SECTION("k beyond the row count returns every row, still sorted")
  {
    auto out = merge_enn_top_k(h.context(/*k=*/100), candidates);
    h.stream.synchronize();

    auto got_ids = to_host<int32_t>(out->get_column(0).view());
    REQUIRE(got_ids == std::vector<int32_t>{3, 1, 2, 0});
  }

  SECTION("k == 0 returns an empty table")
  {
    auto out = merge_enn_top_k(h.context(/*k=*/0), candidates);
    REQUIRE(out->num_rows() == 0);
  }
}
