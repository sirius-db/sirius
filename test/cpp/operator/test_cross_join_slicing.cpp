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

/**
 * @file test_cross_join_slicing.cpp
 * @brief Unit tests for the number of tasks a cross product of two batches is split into.
 */

#include "op/cross_join_slicing.hpp"
#include "op/sirius_physical_nested_loop_join.hpp"

#include <cudf/types.hpp>

#include <catch.hpp>

#include <cstddef>
#include <limits>

using sirius::batch_rows_and_bytes;
using sirius::op::cross_join_num_slices;
using sirius::op::cross_join_output_bytes;

TEST_CASE("cross join slicing - output bytes", "[cross_product][nested_loop_join]")
{
  // Each left byte is repeated once per right row and each right byte once per left row.
  CHECK(cross_join_output_bytes({.rows = 1000, .bytes = 8000}, {.rows = 10, .bytes = 40}) ==
        8000 * 10 + 40 * 1000);
  CHECK(cross_join_output_bytes({.rows = std::nullopt, .bytes = 8000}, {.rows = 10, .bytes = 40}) ==
        0);
  auto constexpr huge = std::numeric_limits<std::size_t>::max() / 2;
  CHECK(cross_join_output_bytes({.rows = huge, .bytes = huge}, {.rows = huge, .bytes = huge}) ==
        std::numeric_limits<std::size_t>::max());
}

TEST_CASE("cross join slicing - output within the budget is one task",
          "[cross_product][nested_loop_join]")
{
  batch_rows_and_bytes const left{.rows = 1000, .bytes = 8000};
  batch_rows_and_bytes const right{.rows = 1000, .bytes = 8000};
  // 16,000,000 output bytes.
  CHECK(cross_join_num_slices(left, right, 16'000'000) == 1);
  CHECK(cross_join_num_slices(left, right, 1'000'000'000) == 1);
}

TEST_CASE("cross join slicing - output over the budget is split and rounded up",
          "[cross_product][nested_loop_join]")
{
  batch_rows_and_bytes const left{.rows = 1000, .bytes = 8000};
  batch_rows_and_bytes const right{.rows = 1000, .bytes = 8000};
  CHECK(cross_join_num_slices(left, right, 15'999'999) == 2);
  CHECK(cross_join_num_slices(left, right, 1'000'000) == 16);
  CHECK(cross_join_num_slices(left, right, 999'999) == 17);
}

TEST_CASE("cross join slicing - at most one task per left row", "[cross_product][nested_loop_join]")
{
  batch_rows_and_bytes const left{.rows = 3, .bytes = 24};
  batch_rows_and_bytes const right{.rows = 1'000'000, .bytes = 8'000'000};
  CHECK(cross_join_num_slices(left, right, 1) == 3);
}

TEST_CASE("cross join slicing - unknown or zero rows are one task",
          "[cross_product][nested_loop_join]")
{
  batch_rows_and_bytes const known{.rows = 1000, .bytes = 8000};
  batch_rows_and_bytes const unknown{.rows = std::nullopt, .bytes = 8000};
  batch_rows_and_bytes const empty{.rows = 0, .bytes = 0};
  CHECK(cross_join_num_slices(unknown, known, 1) == 1);
  CHECK(cross_join_num_slices(known, unknown, 1) == 1);
  CHECK(cross_join_num_slices(empty, known, 1) == 1);
}

TEST_CASE("cross join slicing - each task stays within the cuDF row limit",
          "[cross_product][nested_loop_join]")
{
  // 100,000 x 100,000 = 1e10 output rows need at least 5 tasks of at most 2^31 - 1 rows.
  batch_rows_and_bytes const left{.rows = 100'000, .bytes = 100'000};
  batch_rows_and_bytes const right{.rows = 100'000, .bytes = 100'000};
  SECTION("without a byte budget") { CHECK(cross_join_num_slices(left, right, 0) == 5); }
  SECTION("with a byte budget the output fits")
  {
    CHECK(cross_join_num_slices(left, right, 1ULL << 40) == 5);
  }
}

TEST_CASE("cross join slicing - the row limit counts whole left rows per task",
          "[cross_product][nested_loop_join]")
{
  // 100,001 x 42,949 output rows fit in two tasks by row count alone, but a task holds whole left
  // rows and at most 50,000 of them fit under 2^31 - 1 rows. Three tasks keep every slice within.
  batch_rows_and_bytes const left{.rows = 100'001, .bytes = 100'001};
  batch_rows_and_bytes const right{.rows = 42'949, .bytes = 42'949};
  auto const num_slices = cross_join_num_slices(left, right, 0);
  CHECK(num_slices == 3);
  auto const max_slice_rows = (*left.rows + num_slices - 1) / num_slices;
  CHECK(max_slice_rows * *right.rows <=
        static_cast<std::size_t>(std::numeric_limits<cudf::size_type>::max()));
}

TEST_CASE("cross join slicing - a rebuilt task input keeps its slice",
          "[cross_product][nested_loop_join]")
{
  // Late materialization rebuilds a task input with restored batches. A slice lost there would
  // make the task join the whole pair.
  sirius::op::cross_join_slice_data const input({}, 2, 5);
  auto const rebuilt = input.with_data_batches({});
  auto const* slice  = dynamic_cast<sirius::op::cross_join_slice_data const*>(rebuilt.get());
  REQUIRE(slice != nullptr);
  CHECK(slice->slice == 2);
  CHECK(slice->num_slices == 5);
}
