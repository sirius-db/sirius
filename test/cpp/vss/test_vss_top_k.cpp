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

// Operator-level coverage for the fused VSS kernels. brute_force/interop tests
// cover the cuVS search itself; here we exercise the surrounding
// gather/output-assembly and the map-reduce merge, which the SQL end-to-end
// tests otherwise cover only indirectly.

// test
#include <catch.hpp>

// sirius
#include <vss/enn_top_k.hpp>
#include <vss/vss_pattern.hpp>

// cudf
#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

// rmm
#include <rmm/device_buffer.hpp>

#include <cuda_runtime_api.h>

#include <cstdint>
#include <memory>
#include <vector>

namespace {

std::unique_ptr<cudf::column> make_int32_column(std::vector<int32_t> const& vals)
{
  auto col = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::INT32}, vals.size(), cudf::mask_state::UNALLOCATED);
  cudaMemcpy(col->mutable_view().data<int32_t>(),
             vals.data(),
             sizeof(int32_t) * vals.size(),
             cudaMemcpyHostToDevice);
  return col;
}

std::unique_ptr<cudf::column> make_float32_column(std::vector<float> const& vals)
{
  auto col = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::FLOAT32}, vals.size(), cudf::mask_state::UNALLOCATED);
  cudaMemcpy(col->mutable_view().data<float>(),
             vals.data(),
             sizeof(float) * vals.size(),
             cudaMemcpyHostToDevice);
  return col;
}

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

// Same as make_fixed_size_float_list but with a row-level null mask on the
// parent list. Null rows keep their dim value slots (contents are don't-care).
std::unique_ptr<cudf::column> make_nullable_fixed_size_float_list(
  std::vector<float> const& values,
  cudf::size_type n_rows,
  cudf::size_type dim,
  std::vector<bool> const& row_valid)
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

  std::vector<cudf::bitmask_type> host_mask(cudf::num_bitmask_words(n_rows), 0);
  cudf::size_type null_count = 0;
  for (cudf::size_type i = 0; i < n_rows; ++i) {
    if (row_valid[i]) {
      host_mask[i / 32] |= (cudf::bitmask_type{1} << (i % 32));
    } else {
      ++null_count;
    }
  }
  rmm::device_buffer null_mask(cudf::bitmask_allocation_size_bytes(n_rows),
                               cudf::get_default_stream());
  cudaMemcpy(null_mask.data(),
             host_mask.data(),
             sizeof(cudf::bitmask_type) * host_mask.size(),
             cudaMemcpyHostToDevice);

  return cudf::make_lists_column(
    n_rows, std::move(offsets_col), std::move(child), null_count, std::move(null_mask));
}

std::vector<int32_t> to_host_i32(cudf::column_view const& col)
{
  std::vector<int32_t> host(col.size());
  cudaMemcpy(
    host.data(), col.data<int32_t>(), sizeof(int32_t) * host.size(), cudaMemcpyDeviceToHost);
  return host;
}

std::vector<float> to_host_f(cudf::column_view const& col)
{
  std::vector<float> host(col.size());
  cudaMemcpy(host.data(), col.data<float>(), sizeof(float) * host.size(), cudaMemcpyDeviceToHost);
  return host;
}

// A pattern projecting [id (passthrough), distance], distance last.
sirius::vss::vss_top_k_pattern make_id_distance_pattern(std::vector<float> query)
{
  sirius::vss::vss_top_k_pattern pattern;
  pattern.vector_column_index = 1;  // input column 1 is the vector column
  pattern.dim                 = static_cast<int64_t>(query.size());
  pattern.query               = std::move(query);
  pattern.metric              = cuvs::distance::DistanceType::L2SqrtUnexpanded;
  pattern.output_columns      = {
    {sirius::vss::vss_output_column::kind::gather_input, /*input_index=*/0},
    {sirius::vss::vss_output_column::kind::distance, /*input_index=*/-1},
  };
  pattern.distance_output_index = 1;  // distance is output column 1
  return pattern;
}

}  // namespace

TEST_CASE("compute_vss_top_k gathers passthrough columns with their distances", "[vss]")
{
  auto stream = cudf::get_default_stream();
  auto mr     = cudf::get_current_device_resource_ref();

  // Rows: (id, vec) with Euclidean distance to the origin
  auto ids = make_int32_column({10, 11, 12, 13});
  auto vec = make_fixed_size_float_list(
    {
      3.0f,
      4.0f,  // id 10 -> 5
      1.0f,
      0.0f,  // id 11 -> 1
      0.0f,
      0.0f,  // id 12 -> 0
      0.0f,
      2.0f,  // id 13 -> 2
    },
    /*n_rows=*/4,
    /*dim=*/2);

  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.push_back(std::move(ids));
  cols.push_back(std::move(vec));
  cudf::table input{std::move(cols)};

  auto pattern = make_id_distance_pattern({0.0f, 0.0f});

  SECTION("limit selects the nearest rows, nearest-first")
  {
    auto out =
      sirius::op::compute_vss_top_k(input.view(), pattern, /*limit=*/2, /*offset=*/0, stream, mr);
    stream.synchronize();

    REQUIRE(out->num_columns() == 2);
    REQUIRE(out->num_rows() == 2);
    REQUIRE(to_host_i32(out->view().column(0)) == std::vector<int32_t>{12, 11});
    auto dist = to_host_f(out->view().column(1));
    REQUIRE(dist[0] == Approx(0.0f));
    REQUIRE(dist[1] == Approx(1.0f));
  }

  SECTION("offset widens the kept set (the final OFFSET slice is the merge's job)")
  {
    // keep = offset + limit = 3, so three candidates are returned un-sliced.
    auto out =
      sirius::op::compute_vss_top_k(input.view(), pattern, /*limit=*/2, /*offset=*/1, stream, mr);
    stream.synchronize();

    REQUIRE(out->num_rows() == 3);
    REQUIRE(to_host_i32(out->view().column(0)) == std::vector<int32_t>{12, 11, 13});
    auto dist = to_host_f(out->view().column(1));
    REQUIRE(dist[0] == Approx(0.0f));
    REQUIRE(dist[1] == Approx(1.0f));
    REQUIRE(dist[2] == Approx(2.0f));
  }

  SECTION("limit == 0 yields the empty output schema")
  {
    auto out =
      sirius::op::compute_vss_top_k(input.view(), pattern, /*limit=*/0, /*offset=*/0, stream, mr);
    REQUIRE(out->num_columns() == 2);
    REQUIRE(out->num_rows() == 0);
  }
}

TEST_CASE("compute_vss_top_k drops null rows before the search", "[vss]")
{
  auto stream = cudf::get_default_stream();
  auto mr     = cudf::get_current_device_resource_ref();

  // Row 2 is NULL (its value slots are don't-care). DuckDB would sort the NULL
  // distance last and drop it from the top-k; we drop it up front, same result.
  auto ids = make_int32_column({10, 11, 99, 12, 13});
  auto vec = make_nullable_fixed_size_float_list(
    {
      3.0f,
      4.0f,  // id 10 -> 5
      1.0f,
      0.0f,  // id 11 -> 1
      -1.0f,
      -1.0f,  // id 99 -> NULL row (don't-care)
      0.0f,
      0.0f,  // id 12 -> 0
      0.0f,
      2.0f,  // id 13 -> 2
    },
    /*n_rows=*/5,
    /*dim=*/2,
    /*row_valid=*/{true, true, false, true, true});

  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.push_back(std::move(ids));
  cols.push_back(std::move(vec));
  cudf::table input{std::move(cols)};
  auto pattern = make_id_distance_pattern({0.0f, 0.0f});

  SECTION("nearest rows never include the null row")
  {
    auto out =
      sirius::op::compute_vss_top_k(input.view(), pattern, /*limit=*/3, /*offset=*/0, stream, mr);
    stream.synchronize();

    REQUIRE(out->num_rows() == 3);
    REQUIRE(to_host_i32(out->view().column(0)) == std::vector<int32_t>{12, 11, 13});
    auto dist = to_host_f(out->view().column(1));
    REQUIRE(dist[0] == Approx(0.0f));
    REQUIRE(dist[1] == Approx(1.0f));
    REQUIRE(dist[2] == Approx(2.0f));
  }

  SECTION("fewer non-null rows than the limit returns only the non-null rows")
  {
    // 4 non-null rows but limit=10: we return 4, not 10. DuckDB would pad the
    // tail with the NULL-distance row; dropping it diverges only in this
    // almost-all-null case, which we accept.
    auto out =
      sirius::op::compute_vss_top_k(input.view(), pattern, /*limit=*/10, /*offset=*/0, stream, mr);
    stream.synchronize();

    REQUIRE(out->num_rows() == 4);
    REQUIRE(to_host_i32(out->view().column(0)) == std::vector<int32_t>{12, 11, 13, 10});
  }
}

TEST_CASE("compute_vss_top_k handles a sliced (non-zero offset) input", "[vss]")
{
  auto stream = cudf::get_default_stream();
  auto mr     = cudf::get_current_device_resource_ref();

  // Row 0 is a decoy we slice away; the search must see only rows 1..4. A
  // non-zero offset can't be zero-copy reinterpreted, so the operator compacts.
  auto ids = make_int32_column({77, 10, 11, 12, 13});
  auto vec = make_fixed_size_float_list(
    {
      9.0f,
      9.0f,  // id 77 -> decoy, sliced off
      3.0f,
      4.0f,  // id 10 -> 5
      1.0f,
      0.0f,  // id 11 -> 1
      0.0f,
      0.0f,  // id 12 -> 0
      0.0f,
      2.0f,  // id 13 -> 2
    },
    /*n_rows=*/5,
    /*dim=*/2);

  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.push_back(std::move(ids));
  cols.push_back(std::move(vec));
  cudf::table input{std::move(cols)};

  auto sliced = cudf::slice(input.view(), {1, 5}).front();  // offset = 1 on every column
  REQUIRE(sliced.column(1).offset() != 0);

  auto pattern = make_id_distance_pattern({0.0f, 0.0f});
  auto out = sirius::op::compute_vss_top_k(sliced, pattern, /*limit=*/2, /*offset=*/0, stream, mr);
  stream.synchronize();

  REQUIRE(out->num_rows() == 2);
  REQUIRE(to_host_i32(out->view().column(0)) == std::vector<int32_t>{12, 11});
  auto dist = to_host_f(out->view().column(1));
  REQUIRE(dist[0] == Approx(0.0f));
  REQUIRE(dist[1] == Approx(1.0f));
}

TEST_CASE("merge_vss_top_k consolidates per-batch candidates by distance", "[vss]")
{
  auto stream = cudf::get_default_stream();
  auto mr     = cudf::get_current_device_resource_ref();

  // Concatenated candidates from two batches: [id, distance], distance at idx 1.
  // Batch A -> (12, 0.0), (11, 1.0);  Batch B -> (99, 0.5), (98, 3.0).
  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.push_back(make_int32_column({12, 11, 99, 98}));
  cols.push_back(make_float32_column({0.0f, 1.0f, 0.5f, 3.0f}));
  cudf::table combined{std::move(cols)};

  SECTION("top-k across batches, sorted nearest-first")
  {
    auto out = sirius::op::merge_vss_top_k(
      combined.view(), /*distance_index=*/1, /*limit=*/2, /*offset=*/0, stream, mr);
    stream.synchronize();

    REQUIRE(out->num_rows() == 2);
    REQUIRE(to_host_i32(out->view().column(0)) == std::vector<int32_t>{12, 99});
    auto dist = to_host_f(out->view().column(1));
    REQUIRE(dist[0] == Approx(0.0f));
    REQUIRE(dist[1] == Approx(0.5f));
  }

  SECTION("offset is included in the kept set (sliced later by the operator)")
  {
    auto out = sirius::op::merge_vss_top_k(
      combined.view(), /*distance_index=*/1, /*limit=*/2, /*offset=*/1, stream, mr);
    stream.synchronize();

    REQUIRE(out->num_rows() == 3);
    REQUIRE(to_host_i32(out->view().column(0)) == std::vector<int32_t>{12, 99, 11});
    auto dist = to_host_f(out->view().column(1));
    REQUIRE(dist[0] == Approx(0.0f));
    REQUIRE(dist[1] == Approx(0.5f));
    REQUIRE(dist[2] == Approx(1.0f));
  }

  SECTION("limit == 0 yields an empty table with the same schema")
  {
    auto out = sirius::op::merge_vss_top_k(
      combined.view(), /*distance_index=*/1, /*limit=*/0, /*offset=*/0, stream, mr);
    REQUIRE(out->num_columns() == 2);
    REQUIRE(out->num_rows() == 0);
  }
}
