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
 * @brief Unit tests for splitting a cross product of two batches into tasks.
 */

#include "data/data_batch_utils.hpp"
#include "helper/type_conversions.hpp"
#include "op/cross_join_slicing.hpp"
#include "op/sirius_physical_nested_loop_join.hpp"
#include "operator_test_utils.hpp"

#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <catch.hpp>
#include <duckdb/planner/operator/logical_comparison_join.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

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

TEST_CASE("cross join slicing - a task input rebuilt by late materialization joins only its slice",
          "[cross_product][nested_loop_join][late_mat]")
{
  using sirius::test::operator_utils::copy_column_to_host;
  using sirius::test::operator_utils::make_numeric_batch;
  auto& space       = *sirius::test::operator_utils::get_default_gpu_space();
  auto const stream = cudf::get_default_stream();
  std::vector<std::int32_t> const left_rows{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<std::int32_t> const right_rows{100, 200, 300};

  duckdb::LogicalComparisonJoin logical(duckdb::JoinType::INNER);
  logical.types    = {duckdb::LogicalType::INTEGER, duckdb::LogicalType::INTEGER};
  auto const child = [] {
    return duckdb::make_uniq<sirius::op::sirius_physical_operator>(
      sirius::op::SiriusPhysicalOperatorType::PROJECTION,
      sirius::from_duckdb_vec(duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::INTEGER}),
      0);
  };
  sirius::op::sirius_physical_nested_loop_join join(
    logical, child(), child(), {}, duckdb::JoinType::INNER, 30);

  // The second of two slices: left rows 5..9 with every right row.
  auto const left  = make_numeric_batch(space, left_rows, cudf::type_id::INT32);
  auto const right = make_numeric_batch(space, right_rows, cudf::type_id::INT32);
  sirius::op::cross_join_slice_data const input({left, right}, 1, 2);

  // Late materialization replaces each batch with one holding the same rows.
  auto const rebuilt = input.with_batches_preserving_rows(
    {make_numeric_batch(space, left_rows, cudf::type_id::INT32),
     make_numeric_batch(space, right_rows, cudf::type_id::INT32)});
  auto const* slice = dynamic_cast<sirius::op::cross_join_slice_data const*>(rebuilt.get());
  REQUIRE(slice != nullptr);
  CHECK(slice->slice == 1);
  CHECK(slice->num_slices == 2);
  CHECK(slice->original_batch_ids() ==
        std::vector<std::uint64_t>{left->get_batch_id(), right->get_batch_id()});

  auto const output = join.execute(*rebuilt, stream);
  auto const& batches =
    dynamic_cast<sirius::op::pipelineable_operator_data const&>(*output).get_data_batches();
  REQUIRE(batches.size() == 1);
  auto const view = sirius::get_cudf_table_view(*batches.front());
  REQUIRE(view.num_rows() == 15);
  auto const joined_left = copy_column_to_host<std::int32_t>(view.column(0));
  CHECK(std::ranges::all_of(joined_left, [](std::int32_t row) { return row >= 5; }));
}
