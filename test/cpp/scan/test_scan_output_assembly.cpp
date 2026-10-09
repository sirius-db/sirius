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

#include <cudf/column/column_factories.hpp>
#include <cudf/cudf_utils.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <cuda_runtime_api.h>

#include <catch.hpp>
#include <op/scan/scan_plan.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace scan = sirius::op::scan;

::cuda::stream_ref test_stream() { return cudf::get_default_stream(); }

std::unique_ptr<cudf::column> int32_column(std::vector<std::int32_t> const& values)
{
  auto stream = test_stream();
  auto column = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32},
                                          static_cast<cudf::size_type>(values.size()),
                                          cudf::mask_state::UNALLOCATED,
                                          stream);
  if (!values.empty()) {
    CUDF_CUDA_TRY(cudaMemcpyAsync(column->mutable_view().data<std::int32_t>(),
                                  values.data(),
                                  values.size() * sizeof(std::int32_t),
                                  cudaMemcpyHostToDevice,
                                  stream.get()));
    CUDF_CUDA_TRY(cudaStreamSynchronize(stream.get()));
  }
  return column;
}

std::unique_ptr<cudf::table> int32_table(std::vector<std::vector<std::int32_t>> const& values)
{
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.reserve(values.size());
  for (auto const& column : values) {
    columns.push_back(int32_column(column));
  }
  return std::make_unique<cudf::table>(std::move(columns));
}

std::vector<std::int32_t> to_host(cudf::column_view const& column)
{
  auto stream = test_stream();
  std::vector<std::int32_t> values(static_cast<std::size_t>(column.size()));
  if (!values.empty()) {
    CUDF_CUDA_TRY(cudaMemcpyAsync(values.data(),
                                  column.data<std::int32_t>(),
                                  values.size() * sizeof(std::int32_t),
                                  cudaMemcpyDeviceToHost,
                                  stream.get()));
    CUDF_CUDA_TRY(cudaStreamSynchronize(stream.get()));
  }
  return values;
}

std::vector<void const*> data_pointers(cudf::table_view table)
{
  std::vector<void const*> pointers;
  pointers.reserve(static_cast<std::size_t>(table.num_columns()));
  for (cudf::size_type i = 0; i < table.num_columns(); ++i) {
    pointers.push_back(table.column(i).head());
  }
  return pointers;
}

scan::scan_plan::data_column data_column(std::size_t index)
{
  return {index, "c" + std::to_string(index)};
}

}  // namespace

TEST_CASE("scan output assembly projects and reorders data columns without copying",
          "[scan][parquet][assembly]")
{
  scan::scan_plan plan;
  plan.data_columns  = {data_column(0), data_column(1), data_column(2)};
  plan.output_layout = {{scan::scan_plan::output_entry::DATA, 2},
                        {scan::scan_plan::output_entry::DATA, 0}};

  auto input                   = int32_table({{10, 11}, {20, 21}, {30, 31}});
  auto const original_pointers = data_pointers(input->view());
  scan::owning_table_view view{std::move(input)};

  auto output = scan::assemble_scan_output(plan, std::move(view), {}, test_stream());

  REQUIRE(output.num_rows() == 2);
  REQUIRE(output.n_columns() == 2);
  CHECK(to_host(output.column(0)) == std::vector<std::int32_t>{30, 31});
  CHECK(to_host(output.column(1)) == std::vector<std::int32_t>{10, 11});
  auto const output_pointers = data_pointers(output.view());
  CHECK(output_pointers[0] == original_pointers[2]);
  CHECK(output_pointers[1] == original_pointers[0]);
}

TEST_CASE("scan output assembly copies repeated data columns",
          "[scan][parquet][assembly][duplicate]")
{
  scan::scan_plan plan;
  plan.data_columns  = {data_column(0), data_column(1)};
  plan.output_layout = {{scan::scan_plan::output_entry::DATA, 1},
                        {scan::scan_plan::output_entry::DATA, 0},
                        {scan::scan_plan::output_entry::DATA, 1}};

  auto input                   = int32_table({{10, 11}, {20, 21}});
  auto const original_pointers = data_pointers(input->view());
  scan::owning_table_view view{std::move(input)};

  auto output = scan::assemble_scan_output(plan, std::move(view), {}, test_stream());

  REQUIRE(output.num_rows() == 2);
  REQUIRE(output.n_columns() == 3);
  CHECK(to_host(output.column(0)) == std::vector<std::int32_t>{20, 21});
  CHECK(to_host(output.column(1)) == std::vector<std::int32_t>{10, 11});
  CHECK(to_host(output.column(2)) == std::vector<std::int32_t>{20, 21});

  auto const output_pointers = data_pointers(output.view());
  CHECK(output_pointers[0] == original_pointers[1]);
  CHECK(output_pointers[1] == original_pointers[0]);
  CHECK(output_pointers[2] != output_pointers[0]);
}

TEST_CASE("scan output assembly interleaves partition and repeated data columns",
          "[scan][parquet][assembly][hive][duplicate]")
{
  scan::scan_plan plan;
  plan.data_columns = {data_column(0)};
  plan.partition_columns.push_back(
    {1, "part", sirius::logical_type::make(sirius::type_id::INTEGER)});
  plan.output_layout = {{scan::scan_plan::output_entry::DATA, 0},
                        {scan::scan_plan::output_entry::PARTITION, 0},
                        {scan::scan_plan::output_entry::DATA, 0},
                        {scan::scan_plan::output_entry::PARTITION, 0}};

  auto input                   = int32_table({{10, 11}});
  auto const original_pointers = data_pointers(input->view());
  scan::owning_table_view view{std::move(input)};

  auto output = scan::assemble_scan_output(plan, std::move(view), {"2024"}, test_stream());

  REQUIRE(output.num_rows() == 2);
  REQUIRE(output.n_columns() == 4);
  CHECK(to_host(output.column(0)) == std::vector<std::int32_t>{10, 11});
  CHECK(to_host(output.column(1)) == std::vector<std::int32_t>{2024, 2024});
  CHECK(to_host(output.column(2)) == std::vector<std::int32_t>{10, 11});
  CHECK(to_host(output.column(3)) == std::vector<std::int32_t>{2024, 2024});

  auto const output_pointers = data_pointers(output.view());
  CHECK(output_pointers[0] == original_pointers[0]);
  CHECK(output_pointers[2] != output_pointers[0]);
  CHECK(output_pointers[3] != output_pointers[1]);
}

TEST_CASE("scan output assembly preserves the row-count carrier for an empty layout",
          "[scan][parquet][assembly][carrier]")
{
  scan::scan_plan plan;
  plan.data_columns = {data_column(0)};

  auto input                   = int32_table({{10, 11, 12}});
  auto const original_pointers = data_pointers(input->view());
  scan::owning_table_view view{std::move(input)};

  auto output = scan::assemble_scan_output(plan, std::move(view), {}, test_stream());

  REQUIRE(output.num_rows() == 3);
  REQUIRE(output.n_columns() == 1);
  CHECK(to_host(output.column(0)) == std::vector<std::int32_t>{10, 11, 12});
  CHECK(data_pointers(output.view())[0] == original_pointers[0]);
}

TEST_CASE("scan output assembly routing recognizes pass-through and rebuild layouts",
          "[scan][parquet][assembly][routing]")
{
  scan::scan_plan plan;

  SECTION("empty output layout is a pass-through")
  {
    plan.data_columns = {data_column(0)};
    CHECK_FALSE(scan::needs_output_assembly(plan));
  }

  SECTION("identity data layout is a pass-through")
  {
    plan.data_columns  = {data_column(0), data_column(1)};
    plan.output_layout = {{scan::scan_plan::output_entry::DATA, 0},
                          {scan::scan_plan::output_entry::DATA, 1}};
    CHECK_FALSE(scan::needs_output_assembly(plan));
  }

  SECTION("reordered data layout requires assembly")
  {
    plan.data_columns  = {data_column(0), data_column(1)};
    plan.output_layout = {{scan::scan_plan::output_entry::DATA, 1},
                          {scan::scan_plan::output_entry::DATA, 0}};
    CHECK(scan::needs_output_assembly(plan));
  }

  SECTION("duplicate data layout requires assembly")
  {
    plan.data_columns  = {data_column(0)};
    plan.output_layout = {{scan::scan_plan::output_entry::DATA, 0},
                          {scan::scan_plan::output_entry::DATA, 0}};
    CHECK(scan::needs_output_assembly(plan));
  }

  SECTION("partition layout requires assembly")
  {
    plan.data_columns = {data_column(0)};
    plan.partition_columns.push_back(
      {1, "part", sirius::logical_type::make(sirius::type_id::INTEGER)});
    plan.output_layout = {{scan::scan_plan::output_entry::PARTITION, 0}};
    CHECK(scan::needs_output_assembly(plan));
  }
}

TEST_CASE("scan output prefix width and leading prefix follow the output layout",
          "[scan][parquet][assembly][routing]")
{
  scan::scan_plan plan;

  SECTION("identity with a trailing pure-filter column reads a leading prefix")
  {
    plan.data_columns  = {data_column(0), data_column(1), data_column(2)};
    plan.output_layout = {{scan::scan_plan::output_entry::DATA, 0},
                          {scan::scan_plan::output_entry::DATA, 1}};
    CHECK(scan::output_prefix_width(plan) == std::size_t{2});
    CHECK(scan::output_is_leading_prefix(plan));
  }

  SECTION("plain identity reads every column as its prefix")
  {
    plan.data_columns  = {data_column(0), data_column(1)};
    plan.output_layout = {{scan::scan_plan::output_entry::DATA, 0},
                          {scan::scan_plan::output_entry::DATA, 1}};
    CHECK(scan::output_prefix_width(plan) == std::size_t{2});
    CHECK(scan::output_is_leading_prefix(plan));
  }

  SECTION("reordered output is not a leading prefix")
  {
    plan.data_columns  = {data_column(0), data_column(1), data_column(2)};
    plan.output_layout = {{scan::scan_plan::output_entry::DATA, 1},
                          {scan::scan_plan::output_entry::DATA, 0}};
    CHECK(scan::output_prefix_width(plan) == std::size_t{2});
    CHECK_FALSE(scan::output_is_leading_prefix(plan));
  }

  SECTION("duplicate output reads one column but is not a leading prefix")
  {
    plan.data_columns  = {data_column(0), data_column(1)};
    plan.output_layout = {{scan::scan_plan::output_entry::DATA, 0},
                          {scan::scan_plan::output_entry::DATA, 0}};
    CHECK(scan::output_prefix_width(plan) == std::size_t{1});
    CHECK_FALSE(scan::output_is_leading_prefix(plan));
  }

  SECTION("a synthesized virtual output column is not a leading prefix")
  {
    // DATA(1) sits past the one reader column: it is the first virtual column, which only the
    // assembly synthesizes.
    plan.data_columns = {data_column(0)};
    plan.virtual_columns.push_back({duckdb::column_t{1},
                                    "filename",
                                    sirius::logical_type::make(sirius::type_id::VARCHAR),
                                    scan::scan_plan::parquet_virtual_column_kind::FILENAME,
                                    1});
    plan.output_layout = {{scan::scan_plan::output_entry::DATA, 0},
                          {scan::scan_plan::output_entry::DATA, 1}};
    CHECK(scan::output_prefix_width(plan) == std::size_t{2});
    CHECK_FALSE(scan::output_is_leading_prefix(plan));
  }

  SECTION("a partition output is not a leading prefix")
  {
    plan.data_columns = {data_column(0), data_column(1)};
    plan.partition_columns.push_back(
      {2, "part", sirius::logical_type::make(sirius::type_id::INTEGER)});
    plan.output_layout = {{scan::scan_plan::output_entry::DATA, 0},
                          {scan::scan_plan::output_entry::PARTITION, 0}};
    CHECK(scan::output_prefix_width(plan) == std::size_t{1});
    CHECK_FALSE(scan::output_is_leading_prefix(plan));
  }

  SECTION("an empty layout over a row-count carrier reads no column")
  {
    plan.data_columns        = {data_column(0)};
    plan.carrier_batch_index = 0;
    CHECK_FALSE(scan::output_prefix_width(plan).has_value());
    CHECK_FALSE(scan::output_is_leading_prefix(plan));
  }

  SECTION("a partition-only output reads no column")
  {
    plan.data_columns = {data_column(0)};
    plan.partition_columns.push_back(
      {1, "part", sirius::logical_type::make(sirius::type_id::INTEGER)});
    plan.output_layout = {{scan::scan_plan::output_entry::PARTITION, 0}};
    CHECK_FALSE(scan::output_prefix_width(plan).has_value());
    CHECK_FALSE(scan::output_is_leading_prefix(plan));
  }
}
