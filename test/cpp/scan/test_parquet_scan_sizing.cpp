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

#include <catch.hpp>
#include <io/kvikio/kvikio_context.hpp>
#include <op/scan/parquet_gpu_ingestible.hpp>
#include <op/scan/sirius_gpu_scan_operator_data.hpp>

#include <filesystem>
#include <memory>
#include <vector>

namespace {

namespace scan = sirius::op::scan;

std::filesystem::path project_root()
{
#ifdef SIRIUS_PROJECT_ROOT
  return std::filesystem::path{SIRIUS_PROJECT_ROOT};
#else
  return std::filesystem::current_path();
#endif
}

std::unique_ptr<scan::parquet_ingestible_table_info> make_nation_info(bool pure_filter,
                                                                      bool zero_output = false)
{
  auto info                 = std::make_unique<scan::parquet_ingestible_table_info>();
  info->resolved_file_paths = {
    (project_root() / "test/cpp/integration/data/parquet/nation.parquet").string()};
  info->names = {"n_nationkey", "n_name", "n_regionkey", "n_comment"};
  info->returned_types.push_back(sirius::logical_type::make(sirius::type_id::INTEGER));
  info->returned_types.push_back(sirius::logical_type::make(sirius::type_id::VARCHAR));
  info->returned_types.push_back(sirius::logical_type::make(sirius::type_id::INTEGER));
  info->returned_types.push_back(sirius::logical_type::make(sirius::type_id::VARCHAR));
  if (zero_output) {
    info->column_ids.push_back(duckdb::ColumnIndex(3));
    info->projection_ids = {0};
  } else {
    info->column_ids.push_back(duckdb::ColumnIndex(0));
  }
  if (pure_filter && !zero_output) {
    info->column_ids.push_back(duckdb::ColumnIndex(3));
    info->projection_ids = {0, 1};
  }
  info->scan_output_arity      = zero_output ? 0 : 1;
  info->approximate_batch_size = std::size_t{1} << 30;
  return info;
}

std::unique_ptr<scan::parquet_ingestible_table_info> make_partition_only_nation_info()
{
  auto info               = make_nation_info(true);
  info->column_ids        = {duckdb::ColumnIndex(2), duckdb::ColumnIndex(3)};
  info->projection_ids    = {0, 1};
  info->partition_indices = {duckdb::HivePartitioningIndex("1", 2)};
  return info;
}

struct scan_estimates {
  std::size_t output_bytes;
  std::size_t working_set_bytes;
  std::size_t pure_filter_columns;
};

scan_estimates read_estimates(std::unique_ptr<scan::parquet_ingestible_table_info> info)
{
  auto ingestible = scan::make_ingestible(std::move(info));
  auto ioctx      = std::make_shared<sirius::io::kvikio_context>();
  auto task       = ingestible->next_split_provider(
    [ioctx](std::string_view) -> std::shared_ptr<sirius::io::sirius_ioctx> { return ioctx; });
  REQUIRE(task);

  auto file = task();
  REQUIRE(file);
  auto const file_output  = file->estimated_bytes();
  auto const file_working = file->estimated_working_set_bytes();

  auto coalescer = ingestible->create_batch_coalescer();
  auto splits    = coalescer->push(std::move(file));
  auto tail      = coalescer->flush();
  for (auto& split : tail) {
    splits.push_back(std::move(split));
  }
  REQUIRE(splits.size() == 1);

  auto* split = dynamic_cast<scan::parquet_split_info*>(splits.front().get());
  REQUIRE(split);
  CHECK(split->estimated_bytes() == file_output);
  CHECK(split->estimated_working_set_bytes() == file_working);
  return {split->estimated_bytes(),
          split->estimated_working_set_bytes(),
          split->plan->pure_filter_batch_positions().size()};
}

scan_estimates read_estimates(bool pure_filter, bool zero_output = false)
{
  return read_estimates(make_nation_info(pure_filter, zero_output));
}

}  // namespace

TEST_CASE("parquet batches are capped by decode working set", "[scan][parquet][sizing]")
{
  auto info                    = std::make_unique<scan::parquet_ingestible_table_info>();
  info->approximate_batch_size = 100;
  scan::parquet_gpu_ingestible ingestible{std::move(info)};
  auto coalescer = ingestible.create_batch_coalescer();

  auto file = std::make_unique<scan::parquet_file_scan_info>();
  file->row_groups.push_back({0, 20, 60, 10, 1});
  file->row_groups.push_back({1, 20, 60, 10, 1});

  auto first = coalescer->push(std::move(file));
  auto tail  = coalescer->flush();
  REQUIRE(first.size() == 1);
  REQUIRE(tail.size() == 1);
  for (auto const* batch : {first.front().get(), tail.front().get()}) {
    CHECK(batch->estimated_bytes() == 20);
    CHECK(batch->estimated_working_set_bytes() == 60);
  }
  scan::scan_operator_input input{std::move(first.front())};
  CHECK(input.get_estimated_size_in_bytes() == 20);
  CHECK(input.get_estimated_working_set_size_in_bytes() == 60);
}

TEST_CASE("parquet synthetic filter-only columns only increase the decode working set",
          "[scan][parquet][sizing]")
{
  auto const output_only = read_estimates(false);
  auto const with_filter = read_estimates(true);

  REQUIRE(output_only.pure_filter_columns == 0);
  REQUIRE(with_filter.pure_filter_columns == 1);
  CHECK(output_only.working_set_bytes == output_only.output_bytes);
  CHECK(with_filter.output_bytes == output_only.output_bytes);
  CHECK(with_filter.working_set_bytes > output_only.working_set_bytes);
}

TEST_CASE("parquet zero-output scans account for the retained filter column",
          "[scan][parquet][sizing]")
{
  auto const filter_only = read_estimates(true, true);

  REQUIRE(filter_only.pure_filter_columns == 1);
  CHECK(filter_only.output_bytes > 0);
  CHECK(filter_only.working_set_bytes == filter_only.output_bytes);
}

TEST_CASE("parquet partition-only scans keep a nonzero history basis", "[scan][parquet][sizing]")
{
  auto const partition_only = read_estimates(make_partition_only_nation_info());

  REQUIRE(partition_only.pure_filter_columns == 1);
  CHECK(partition_only.output_bytes > 0);
  CHECK(partition_only.working_set_bytes == partition_only.output_bytes);
}
