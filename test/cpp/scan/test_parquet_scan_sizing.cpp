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
#include <duckdb/common/constants.hpp>
#include <io/kvikio/kvikio_context.hpp>
#include <op/scan/parquet_gpu_ingestible.hpp>
#include <op/scan/scan_plan.hpp>
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
    [ioctx](std::string_view) -> std::shared_ptr<sirius::io::ioctx> { return ioctx; });
  REQUIRE(task);

  auto file = task();
  REQUIRE(file);
  auto const file_output  = file->estimated_bytes();
  auto const file_working = file->estimated_working_set_bytes();

  auto coalescer = ingestible->create_batch_coalescer();
  auto batches   = coalescer->push(std::move(file));
  auto tail      = coalescer->flush();
  for (auto& batch : tail) {
    batches.push_back(std::move(batch));
  }
  REQUIRE(batches.size() == 1);

  auto* batch = dynamic_cast<scan::parquet_split_info*>(batches.front().get());
  REQUIRE(batch);
  CHECK(batch->estimated_bytes() == file_output);
  CHECK(batch->estimated_working_set_bytes() == file_working);
  return {batch->estimated_bytes(),
          batch->estimated_working_set_bytes(),
          batch->plan->pure_filter_batch_positions().size()};
}

scan_estimates read_estimates(bool pure_filter, bool zero_output = false)
{
  return read_estimates(make_nation_info(pure_filter, zero_output));
}

duckdb::vector<std::string> plan_names()
{
  return {"n_nationkey", "n_name", "n_regionkey", "n_comment", "year"};
}

duckdb::vector<std::string> data_plan_names()
{
  return {"n_nationkey", "n_name", "n_regionkey", "n_comment"};
}

duckdb::vector<sirius::logical_type> plan_types()
{
  return {sirius::logical_type::make(sirius::type_id::INTEGER),
          sirius::logical_type::make(sirius::type_id::VARCHAR),
          sirius::logical_type::make(sirius::type_id::INTEGER),
          sirius::logical_type::make(sirius::type_id::VARCHAR),
          sirius::logical_type::make(sirius::type_id::INTEGER)};
}

duckdb::vector<sirius::logical_type> data_plan_types()
{
  return {sirius::logical_type::make(sirius::type_id::INTEGER),
          sirius::logical_type::make(sirius::type_id::VARCHAR),
          sirius::logical_type::make(sirius::type_id::INTEGER),
          sirius::logical_type::make(sirius::type_id::VARCHAR)};
}

duckdb::vector<duckdb::HivePartitioningIndex> year_partition()
{
  return {duckdb::HivePartitioningIndex("2024", 4)};
}

}  // namespace

TEST_CASE("parquet scans without a prefetch cache retain advisory ranges",
          "[scan][parquet][prefetch]")
{
  auto ingestible = scan::make_ingestible(make_nation_info(false));
  auto ioctx      = std::make_shared<sirius::io::kvikio_context>();
  REQUIRE_FALSE(ioctx->uses_prefetching_cache());
  auto task = ingestible->next_split_provider(
    [ioctx](std::string_view) -> std::shared_ptr<sirius::io::ioctx> { return ioctx; });
  REQUIRE(task);

  auto coalescer = ingestible->create_batch_coalescer();
  auto batches   = coalescer->push(task());
  auto tail      = coalescer->flush();
  for (auto& batch : tail) {
    batches.push_back(std::move(batch));
  }
  REQUIRE(batches.size() == 1);

  auto* batch = dynamic_cast<scan::parquet_split_info*>(batches.front().get());
  REQUIRE(batch);
  auto const hints = batch->fadvise_hints();
  REQUIRE(hints.size() == 1);
  CHECK(hints.front().datasource != nullptr);
  CHECK_FALSE(hints.front().ranges.empty());
}

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

  // A keep-masked metadata split adds the filter-by-copy envelope on top of
  // the decode working set: compacted output (input-bounded) + the BOOL8
  // expansion (1 B/row) + the uploaded mask words.
  scan::scan_operator_input masked{std::move(tail.front())};
  constexpr std::size_t rows = 40;
  auto words = std::make_shared<std::vector<std::uint32_t>>((rows + 31) / 32, 0xFFFFFFFFu);
  masked.mvcc_keep_mask = sirius::scan_manager::mvcc_chunk_mask{{words, words->data()}, rows};
  CHECK(masked.get_estimated_working_set_size_in_bytes() ==
        2 * 60 + rows + masked.mvcc_keep_mask.view().size_bytes());
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

TEST_CASE("parquet scan plan avoids empty reader projection for hive count star",
          "[scan][parquet][hive][scan_plan]")
{
  auto const names = plan_names();
  auto const types = plan_types();

  SECTION("virtual-only count star with hive partitions is not reader-projected")
  {
    duckdb::vector<duckdb::ColumnIndex> column_ids{
      duckdb::ColumnIndex(duckdb::COLUMN_IDENTIFIER_ROW_ID)};
    duckdb::vector<duckdb::idx_t> projection_ids;

    auto plan = scan::build_scan_plan(column_ids,
                                      projection_ids,
                                      names,
                                      types,
                                      /*output_types_size=*/0,
                                      year_partition());

    CHECK_FALSE(plan.needs_reader_projection);
    CHECK_FALSE(plan.is_projected());
    CHECK(plan.data_columns.empty());
    CHECK(plan.output_layout.empty());
    CHECK_FALSE(plan.has_partitions());
  }

  SECTION("count star with a retained real filter column still projects the reader")
  {
    duckdb::vector<duckdb::ColumnIndex> column_ids{duckdb::ColumnIndex(3)};
    duckdb::vector<duckdb::idx_t> projection_ids{0};

    auto plan = scan::build_scan_plan(column_ids,
                                      projection_ids,
                                      names,
                                      types,
                                      /*output_types_size=*/0,
                                      duckdb::vector<duckdb::HivePartitioningIndex>{});

    CHECK(plan.needs_reader_projection);
    CHECK(plan.is_projected());
    REQUIRE(plan.data_columns.size() == 1);
    CHECK(plan.data_columns[0].primary_idx == 3);
    CHECK(plan.output_layout.empty());
  }

  SECTION("partition-only output injects partitions without projecting an empty data read")
  {
    duckdb::vector<duckdb::ColumnIndex> column_ids{duckdb::ColumnIndex(4)};
    duckdb::vector<duckdb::idx_t> projection_ids;

    auto plan = scan::build_scan_plan(column_ids,
                                      projection_ids,
                                      names,
                                      types,
                                      /*output_types_size=*/1,
                                      year_partition());

    CHECK_FALSE(plan.needs_reader_projection);
    CHECK_FALSE(plan.is_projected());
    CHECK(plan.data_columns.empty());
    REQUIRE(plan.partition_columns.size() == 1);
    CHECK(plan.partition_columns[0].primary_idx == 4);
    CHECK(plan.partition_columns[0].name == "year");
    CHECK(plan.has_partitions());
    REQUIRE(plan.output_layout.size() == 1);
    CHECK(plan.output_layout[0].source == scan::scan_plan::output_entry::PARTITION);
  }

  SECTION("select star over data columns remains an unprojected identity read")
  {
    auto const data_names = data_plan_names();
    auto const data_types = data_plan_types();
    duckdb::vector<duckdb::ColumnIndex> column_ids{duckdb::ColumnIndex(0),
                                                   duckdb::ColumnIndex(1),
                                                   duckdb::ColumnIndex(2),
                                                   duckdb::ColumnIndex(3)};
    duckdb::vector<duckdb::idx_t> projection_ids;

    auto plan = scan::build_scan_plan(column_ids,
                                      projection_ids,
                                      data_names,
                                      data_types,
                                      /*output_types_size=*/4,
                                      duckdb::vector<duckdb::HivePartitioningIndex>{});

    CHECK_FALSE(plan.needs_reader_projection);
    CHECK_FALSE(plan.is_projected());
    REQUIRE(plan.data_columns.size() == 4);
    REQUIRE(plan.output_layout.size() == 4);
    CHECK_FALSE(plan.has_partitions());
  }
}
