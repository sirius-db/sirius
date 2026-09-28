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
