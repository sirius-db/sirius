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

// Tests for the partition_inject_fn_t plumbing introduced when parquet_split_provider
// gained multi-file split bundling. The closures used to parse the file path on every
// invocation; they now consume a pre-computed partition_values vector instead. These
// tests verify that contract end-to-end on the GPU using cudf, plus a smoke test for
// the parquet_scan_data plumbing.

// test
#include <catch.hpp>

// sirius
#include <op/scan/hive_partition.hpp>
#include <op/scan/parquet_scan_operator_data.hpp>
#include <op/scan/scan_plan.hpp>

// duckdb
#include <duckdb/common/column_index.hpp>
#include <duckdb/common/types.hpp>
#include <duckdb/common/vector.hpp>

// cudf
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

// cuda
#include <cuda_runtime_api.h>

// standard library
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

namespace sscan = sirius::op::scan;

namespace {

auto const k_stream = cudf::get_default_stream();
auto const k_mr     = cudf::get_current_device_resource_ref();

//===----------------------------------------------------------------------===//
// Cudf table-building helpers
//===----------------------------------------------------------------------===//

std::unique_ptr<cudf::column> make_int32_column(std::vector<int32_t> const& values)
{
  auto const n = static_cast<cudf::size_type>(values.size());
  auto col     = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::INT32}, n, cudf::mask_state::UNALLOCATED, k_stream, k_mr);
  if (n > 0) {
    cudaMemcpy(col->mutable_view().data<int32_t>(),
               values.data(),
               sizeof(int32_t) * n,
               cudaMemcpyHostToDevice);
  }
  return col;
}

std::unique_ptr<cudf::table> make_table(std::vector<std::vector<int32_t>> const& cols_data)
{
  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.reserve(cols_data.size());
  for (auto const& v : cols_data) {
    cols.push_back(make_int32_column(v));
  }
  return std::make_unique<cudf::table>(std::move(cols));
}

std::vector<int32_t> copy_int32_column(cudf::column_view const& col)
{
  std::vector<int32_t> host(col.size());
  if (col.size() > 0) {
    cudaMemcpy(
      host.data(), col.data<int32_t>(), sizeof(int32_t) * col.size(), cudaMemcpyDeviceToHost);
  }
  return host;
}

//===----------------------------------------------------------------------===//
// scan_plan builders
//===----------------------------------------------------------------------===//

sscan::scan_plan::data_column data_col(std::size_t primary_idx, std::string name)
{
  return {primary_idx, std::move(name)};
}

sscan::scan_plan::partition_column partition_col(std::size_t primary_idx,
                                                 std::string name,
                                                 sirius::type_id tid)
{
  return {primary_idx, std::move(name), sirius::logical_type::make(tid)};
}

sscan::scan_plan::output_entry data_entry(std::size_t idx)
{
  return {sscan::scan_plan::output_entry::DATA, idx};
}

sscan::scan_plan::output_entry partition_entry(std::size_t idx)
{
  return {sscan::scan_plan::output_entry::PARTITION, idx};
}

}  // namespace

//===----------------------------------------------------------------------===//
// scan_plan::build_inject_fn — null-closure cases
//===----------------------------------------------------------------------===//

TEST_CASE("scan_plan::build_inject_fn returns null when output_layout is empty",
          "[scan][partition_inject]")
{
  // SELECT count(*) shape: no output columns. The closure must be null so the
  // reader's row-count-bearing batch flows through untouched.
  sscan::scan_plan plan;
  plan.data_columns      = {data_col(0, "a")};
  plan.partition_columns = {};
  plan.output_layout     = {};
  REQUIRE(plan.build_inject_fn() == nullptr);
}

TEST_CASE("scan_plan::build_inject_fn returns null on identity layout", "[scan][partition_inject]")
{
  // No partitions, output_layout is 1:1 over data_columns in order — the reader's
  // natural output is what downstream wants. No reshaping needed.
  sscan::scan_plan plan;
  plan.data_columns      = {data_col(0, "a"), data_col(1, "b")};
  plan.partition_columns = {};
  plan.output_layout     = {data_entry(0), data_entry(1)};
  REQUIRE(plan.build_inject_fn() == nullptr);
}

//===----------------------------------------------------------------------===//
// scan_plan::build_inject_fn — partition value injection
//===----------------------------------------------------------------------===//

TEST_CASE("scan_plan::build_inject_fn injects integer partition column from values vector",
          "[scan][partition_inject]")
{
  // 1 data column + 1 partition column. partition_values supplies the constant value
  // for the partition column; the closure should turn it into a per-row constant column.
  sscan::scan_plan plan;
  plan.data_columns      = {data_col(0, "a")};
  plan.partition_columns = {partition_col(1, "year", sirius::type_id::INTEGER)};
  plan.output_layout     = {data_entry(0), partition_entry(0)};

  auto fn = plan.build_inject_fn();
  REQUIRE(fn != nullptr);

  auto input  = make_table({{10, 20, 30, 40}});
  auto output = fn(std::move(input), {"2024"}, k_stream);
  REQUIRE(output != nullptr);
  REQUIRE(output->num_columns() == 2);
  REQUIRE(output->num_rows() == 4);

  auto data_vals = copy_int32_column(output->view().column(0));
  REQUIRE(data_vals == std::vector<int32_t>{10, 20, 30, 40});

  REQUIRE(output->view().column(1).type().id() == cudf::type_id::INT32);
  auto part_vals = copy_int32_column(output->view().column(1));
  REQUIRE(part_vals == std::vector<int32_t>{2024, 2024, 2024, 2024});
}

TEST_CASE("scan_plan::build_inject_fn places partition column at any output position",
          "[scan][partition_inject]")
{
  // PARTITION may appear before or interleaved with DATA in output_layout.
  // Verify the closure honors the requested position rather than appending.
  sscan::scan_plan plan;
  plan.data_columns      = {data_col(0, "a")};
  plan.partition_columns = {partition_col(1, "year", sirius::type_id::INTEGER)};
  plan.output_layout     = {partition_entry(0), data_entry(0)};

  auto fn = plan.build_inject_fn();
  REQUIRE(fn != nullptr);

  auto input  = make_table({{7, 8, 9}});
  auto output = fn(std::move(input), {"1999"}, k_stream);
  REQUIRE(output->num_columns() == 2);

  // First column is the partition (constant), second is the data.
  auto first  = copy_int32_column(output->view().column(0));
  auto second = copy_int32_column(output->view().column(1));
  REQUIRE(first == std::vector<int32_t>{1999, 1999, 1999});
  REQUIRE(second == std::vector<int32_t>{7, 8, 9});
}

TEST_CASE("scan_plan::build_inject_fn drops pure-filter data columns absent from output_layout",
          "[scan][partition_inject]")
{
  // data_columns has 2 columns ("a", "b"), but output_layout only references "a".
  // "b" was read for filter evaluation only; the closure must drop it from the result.
  sscan::scan_plan plan;
  plan.data_columns      = {data_col(0, "a"), data_col(1, "b")};
  plan.partition_columns = {};
  plan.output_layout     = {data_entry(0)};

  auto fn = plan.build_inject_fn();
  REQUIRE(fn != nullptr);

  auto input  = make_table({{1, 2, 3}, {100, 200, 300}});
  auto output = fn(std::move(input), {}, k_stream);
  REQUIRE(output->num_columns() == 1);
  REQUIRE(copy_int32_column(output->view().column(0)) == std::vector<int32_t>{1, 2, 3});
}

TEST_CASE("scan_plan::build_inject_fn supports two partitions indexed by partition_values order",
          "[scan][partition_inject]")
{
  // Two partition columns: closure should index partition_values directly with entry.idx
  // (which is the index into partition_columns). Verify both come through correctly.
  sscan::scan_plan plan;
  plan.data_columns      = {data_col(0, "a")};
  plan.partition_columns = {partition_col(1, "year", sirius::type_id::INTEGER),
                            partition_col(2, "month", sirius::type_id::INTEGER)};
  plan.output_layout     = {data_entry(0), partition_entry(0), partition_entry(1)};

  auto fn = plan.build_inject_fn();
  REQUIRE(fn != nullptr);

  auto input  = make_table({{1, 2}});
  auto output = fn(std::move(input), {"2024", "11"}, k_stream);
  REQUIRE(output->num_columns() == 3);
  REQUIRE(copy_int32_column(output->view().column(1)) == std::vector<int32_t>{2024, 2024});
  REQUIRE(copy_int32_column(output->view().column(2)) == std::vector<int32_t>{11, 11});
}

TEST_CASE("scan_plan::build_inject_fn throws when partition_values shorter than required",
          "[scan][partition_inject]")
{
  // Caller bug: closure expects 2 partition values but only 1 is provided. Must throw,
  // not silently read past the end.
  sscan::scan_plan plan;
  plan.data_columns      = {};
  plan.partition_columns = {partition_col(0, "year", sirius::type_id::INTEGER),
                            partition_col(1, "month", sirius::type_id::INTEGER)};
  plan.output_layout     = {partition_entry(0), partition_entry(1)};

  auto fn = plan.build_inject_fn();
  REQUIRE(fn != nullptr);

  auto input = std::make_unique<cudf::table>(std::vector<std::unique_ptr<cudf::column>>{});
  REQUIRE_THROWS(fn(std::move(input), {"2024"}, k_stream));
}

TEST_CASE("scan_plan::build_inject_fn handles VARCHAR partition columns",
          "[scan][partition_inject]")
{
  // Non-numeric partition: closure must produce a STRING-typed constant column.
  sscan::scan_plan plan;
  plan.data_columns      = {data_col(0, "a")};
  plan.partition_columns = {partition_col(1, "region", sirius::type_id::VARCHAR)};
  plan.output_layout     = {data_entry(0), partition_entry(0)};

  auto fn = plan.build_inject_fn();
  REQUIRE(fn != nullptr);

  auto input  = make_table({{1, 2, 3}});
  auto output = fn(std::move(input), {"us-west"}, k_stream);
  REQUIRE(output->num_columns() == 2);
  REQUIRE(output->view().column(1).type().id() == cudf::type_id::STRING);
  REQUIRE(output->view().column(1).size() == 3);
}

//===----------------------------------------------------------------------===//
// build_partition_inject_fn (legacy) — partition_values_idx mapping
//===----------------------------------------------------------------------===//

TEST_CASE("build_partition_inject_fn injects partition values by precomputed index",
          "[scan][partition_inject][legacy]")
{
  // Legacy closure builds a partition_name → idx map at construction so it can index the
  // partition_values vector at call time without a runtime name lookup. Verify the mapping
  // is used correctly when partition columns appear in column_ids in a different order than
  // hive_partition_columns.

  // Schema: column 0 = "month", column 1 = "data", column 2 = "year"
  // hive partition columns are "month" (P=0) and "year" (P=2).
  // Pass them to build_partition_inject_fn in order [year, month] — so partition_values index
  // 0 corresponds to "year" and index 1 to "month". The closure should still pull the right
  // value for each partition_name.
  duckdb::vector<duckdb::ColumnIndex> column_ids;
  column_ids.emplace_back(0);  // month
  column_ids.emplace_back(1);  // data
  column_ids.emplace_back(2);  // year

  duckdb::vector<std::string> names{"month", "data", "year"};
  duckdb::vector<sirius::logical_type> returned_types{
    sirius::logical_type::make(sirius::type_id::INTEGER),
    sirius::logical_type::make(sirius::type_id::INTEGER),
    sirius::logical_type::make(sirius::type_id::INTEGER)};

  std::vector<size_t> selected_column_indices{1};  // only the data column reaches the reader

  std::vector<sscan::hive_partition_column> hive_partition_columns{
    {"year", 2},   // partition_values index 0
    {"month", 0},  // partition_values index 1
  };
  std::unordered_set<size_t> hive_partition_index_set{0, 2};

  auto fn = sirius::build_partition_inject_fn(column_ids,
                                              names,
                                              returned_types,
                                              selected_column_indices,
                                              hive_partition_columns,
                                              hive_partition_index_set);
  REQUIRE(fn != nullptr);

  // partition_values follows hive_partition_columns order: [year, month] = [2024, 11].
  // file_path is required by the legacy typedef but ignored by this closure.
  auto input = make_table({{42, 43}});
  auto output =
    fn(std::move(input), "/data/year=2024/month=11/p.parquet", {"2024", "11"}, k_stream);

  // Output is in column_ids order: month, data, year.
  REQUIRE(output->num_columns() == 3);
  REQUIRE(output->num_rows() == 2);
  REQUIRE(copy_int32_column(output->view().column(0)) == std::vector<int32_t>{11, 11});
  REQUIRE(copy_int32_column(output->view().column(1)) == std::vector<int32_t>{42, 43});
  REQUIRE(copy_int32_column(output->view().column(2)) == std::vector<int32_t>{2024, 2024});
}

TEST_CASE("build_partition_inject_fn throws when partition_values too short",
          "[scan][partition_inject][legacy]")
{
  duckdb::vector<duckdb::ColumnIndex> column_ids;
  column_ids.emplace_back(0);  // year

  duckdb::vector<std::string> names{"year"};
  duckdb::vector<sirius::logical_type> returned_types{
    sirius::logical_type::make(sirius::type_id::INTEGER)};
  std::vector<size_t> selected_column_indices;
  std::vector<sscan::hive_partition_column> hive_partition_columns{{"year", 0}};
  std::unordered_set<size_t> hive_partition_index_set{0};

  auto fn = sirius::build_partition_inject_fn(column_ids,
                                              names,
                                              returned_types,
                                              selected_column_indices,
                                              hive_partition_columns,
                                              hive_partition_index_set);
  REQUIRE(fn != nullptr);

  auto input = std::make_unique<cudf::table>(std::vector<std::unique_ptr<cudf::column>>{});
  REQUIRE_THROWS(fn(std::move(input), "/data/year=2024/p.parquet", {}, k_stream));
}

//===----------------------------------------------------------------------===//
// parquet_scan_data — partition_values plumbing
//===----------------------------------------------------------------------===//

TEST_CASE("parquet_scan_data preserves partition_values through construction",
          "[scan][partition_inject]")
{
  // Smoke test the new field on parquet_scan_data: the split provider populates it once
  // per split (per hive bucket), and the GPU scan operator hands it straight to the
  // partition_inject_fn. Verify it round-trips faithfully.
  std::vector<sscan::row_group_slice> slices;  // empty is fine for a constructor smoke test
  auto reader_options = std::make_shared<cudf::io::parquet_reader_options>(
    cudf::io::parquet_reader_options::builder().build());
  std::shared_ptr<duckdb::Expression> filter;
  auto plan = std::make_shared<sscan::scan_plan const>();

  std::vector<std::string> partition_values{"2024", "11", "us-west"};

  sscan::parquet_scan_data data(std::move(slices), reader_options, filter, plan, partition_values);

  REQUIRE(data.partition_values == std::vector<std::string>{"2024", "11", "us-west"});
}

TEST_CASE("parquet_scan_data tolerates empty partition_values", "[scan][partition_inject]")
{
  // No-hive-partitions case: empty vector is the canonical "all files bundleable" marker
  // emitted by parquet_split_provider when scan_plan::partition_columns is empty.
  std::vector<sscan::row_group_slice> slices;
  auto reader_options = std::make_shared<cudf::io::parquet_reader_options>(
    cudf::io::parquet_reader_options::builder().build());
  std::shared_ptr<duckdb::Expression> filter;
  auto plan = std::make_shared<sscan::scan_plan const>();

  sscan::parquet_scan_data data(std::move(slices), reader_options, filter, plan, {});

  REQUIRE(data.partition_values.empty());
}
