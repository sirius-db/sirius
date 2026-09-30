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
#include <vss/cudf_raft_interop.hpp>

// cudf
#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/types.hpp>

// rmm
#include <rmm/device_buffer.hpp>

#include <cuda_runtime_api.h>

#include <cstdint>
#include <numeric>
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

// Same layout as make_fixed_size_float_list, but with parent row `null_row`
// marked invalid so the column reports a non-zero null_count.
std::unique_ptr<cudf::column> make_list_with_null_parent_row(std::vector<float> const& values,
                                                             cudf::size_type n_rows,
                                                             cudf::size_type dim,
                                                             cudf::size_type null_row)
{
  auto contents = make_fixed_size_float_list(values, n_rows, dim)->release();
  auto offsets  = std::move(contents.children[0]);
  auto child    = std::move(contents.children[1]);

  auto mask = cudf::create_null_mask(n_rows, cudf::mask_state::ALL_VALID);
  cudf::set_null_mask(static_cast<cudf::bitmask_type*>(mask.data()), null_row, null_row + 1, false);

  return cudf::make_lists_column(n_rows, std::move(offsets), std::move(child), 1, std::move(mask));
}

// Same layout as make_fixed_size_float_list, but with a single *element*
// (child value at flat index `null_element`) marked invalid.
std::unique_ptr<cudf::column> make_list_with_null_child_element(std::vector<float> const& values,
                                                                cudf::size_type n_rows,
                                                                cudf::size_type dim,
                                                                cudf::size_type null_element)
{
  auto contents = make_fixed_size_float_list(values, n_rows, dim)->release();
  auto offsets  = std::move(contents.children[0]);
  auto child    = std::move(contents.children[1]);

  auto child_mask = cudf::create_null_mask(child->size(), cudf::mask_state::ALL_VALID);
  cudf::set_null_mask(
    static_cast<cudf::bitmask_type*>(child_mask.data()), null_element, null_element + 1, false);
  child->set_null_mask(std::move(child_mask), 1);

  // Parent rows all valid (null_count 0); the null lives in the values child.
  return cudf::make_lists_column(
    n_rows, std::move(offsets), std::move(child), 0, rmm::device_buffer{});
}

}  // namespace

TEST_CASE("list_column_as_dataset_view wraps a FLOAT[dim] column zero-copy", "[vss]")
{
  constexpr cudf::size_type n_rows = 5;
  constexpr cudf::size_type dim    = 3;

  std::vector<float> values(n_rows * dim);
  std::iota(values.begin(), values.end(), 1.0f);  // 1, 2, 3, ... 15

  auto list_col = make_fixed_size_float_list(values, n_rows, dim);
  auto view     = sirius::vss::list_column_as_dataset_view(list_col->view(), dim);

  SECTION("extents reflect [n_rows, dim]")
  {
    REQUIRE(view.extent(0) == n_rows);
    REQUIRE(view.extent(1) == dim);
  }

  SECTION("view borrows the values child (no copy)")
  {
    REQUIRE(view.data_handle() == list_col->view().child(1).data<float>());
  }

  SECTION("values round-trip correctly through the view")
  {
    std::vector<float> roundtrip(values.size());
    cudaMemcpy(roundtrip.data(),
               view.data_handle(),
               sizeof(float) * roundtrip.size(),
               cudaMemcpyDeviceToHost);
    REQUIRE(roundtrip == values);
  }
}

TEST_CASE("list_column_as_dataset_view rejects malformed input", "[vss]")
{
  constexpr cudf::size_type n_rows = 4;
  constexpr cudf::size_type dim    = 2;
  std::vector<float> values(n_rows * dim, 0.0f);
  auto list_col = make_fixed_size_float_list(values, n_rows, dim);

  SECTION("non-LIST column throws")
  {
    auto plain = cudf::make_numeric_column(
      cudf::data_type{cudf::type_id::FLOAT32}, n_rows, cudf::mask_state::UNALLOCATED);
    REQUIRE_THROWS(sirius::vss::list_column_as_dataset_view(plain->view(), dim));
  }

  SECTION("dim mismatch throws")
  {
    REQUIRE_THROWS(sirius::vss::list_column_as_dataset_view(list_col->view(), dim + 1));
  }

  SECTION("non-positive dim throws")
  {
    REQUIRE_THROWS(sirius::vss::list_column_as_dataset_view(list_col->view(), 0));
  }

  SECTION("null array rows throw (null handling is deferred)")
  {
    auto with_null = make_list_with_null_parent_row(values, n_rows, dim, /*null_row=*/1);
    REQUIRE(with_null->null_count() == 1);
    REQUIRE_THROWS(sirius::vss::list_column_as_dataset_view(with_null->view(), dim));
  }

  SECTION("null vector elements throw (element-level nulls rejected)")
  {
    // Parent rows are all valid, so this survives the row-level compaction in
    // compute_enn_top_k; the null hides in the values child and must be caught
    // here (RAFT reads the buffer as a raw blob and can't skip null slots).
    auto with_null = make_list_with_null_child_element(values, n_rows, dim, /*null_element=*/3);
    REQUIRE(with_null->null_count() == 0);
    REQUIRE(with_null->view().child(1).null_count() == 1);
    REQUIRE_THROWS(sirius::vss::list_column_as_dataset_view(with_null->view(), dim));
  }

  SECTION("sliced column (non-zero offset) throws")
  {
    // Dropping the first row leaves a parent offset of 1, so a flat
    // reinterpretation of the values buffer would start at the wrong vector.
    auto sliced = cudf::slice(list_col->view(), {1, n_rows}).front();
    REQUIRE(sliced.offset() != 0);
    REQUIRE_THROWS(sirius::vss::list_column_as_dataset_view(sliced, dim));
  }
}
