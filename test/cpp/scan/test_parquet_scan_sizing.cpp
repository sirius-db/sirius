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
#include <duckdb.hpp>
#include <duckdb/common/constants.hpp>
#include <io/kvikio/kvikio_context.hpp>
#include <op/scan/parquet_gpu_ingestible.hpp>
#include <op/scan/scan_plan.hpp>
#include <op/scan/sirius_gpu_scan_operator_data.hpp>
#include <utils/parquet_fixture_utils.hpp>

#include <algorithm>
#include <filesystem>
#include <limits>
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

duckdb::vector<std::string> carrier_names()
{
  return {"id", "amount", "label", "small", "day", "flag"};
}

duckdb::vector<sirius::logical_type> carrier_types()
{
  return {sirius::logical_type::make(sirius::type_id::BIGINT),
          sirius::logical_type::make(sirius::type_id::DOUBLE),
          sirius::logical_type::make(sirius::type_id::VARCHAR),
          sirius::logical_type::make(sirius::type_id::SMALLINT),
          sirius::logical_type::make(sirius::type_id::DATE),
          sirius::logical_type::make(sirius::type_id::BOOLEAN)};
}

void check_unmapped_carrier(scan::scan_plan const& plan,
                            duckdb::vector<std::string> const& names,
                            duckdb::vector<sirius::logical_type> const& types)
{
  CHECK(plan.needs_reader_projection);
  CHECK(plan.is_projected());
  REQUIRE(plan.data_columns.size() == 1);
  auto const primary = plan.data_columns.front().primary_idx;
  REQUIRE(primary < types.size());
  CHECK_FALSE(duckdb::IsVirtualColumn(primary));
  CHECK(plan.partition_primary_indices.count(primary) == 0);
  REQUIRE(types[primary].is_fixed_width());
  CHECK(plan.data_columns.front().name == names.at(primary));
  auto const width = types[primary].fixed_width_byte_size();
  for (std::size_t i = 0; i < types.size(); ++i) {
    if (!types[i].is_fixed_width() || plan.partition_primary_indices.count(i)) { continue; }
    CHECK(width <= types[i].fixed_width_byte_size());
    if (width == types[i].fixed_width_byte_size()) { CHECK(primary <= i); }
  }
  for (auto const& position : plan.batch_position_by_column_id) {
    CHECK_FALSE(position.has_value());
  }
}

struct carrier_file_fixture {
  sirius::test::scratch_dir scratch{"parquet_carrier_sizing"};
  std::shared_ptr<sirius::io::kvikio_context> ioctx =
    std::make_shared<sirius::io::kvikio_context>();

  carrier_file_fixture()
  {
    sirius::test::scoped_sirius_disable disable;
    duckdb::DuckDB db(nullptr);
    duckdb::Connection con(db);
    auto run = [&](std::string const& sql) {
      auto result = con.Query(sql);
      REQUIRE(result);
      if (result->HasError()) { UNSCOPED_INFO(result->GetError()); }
      REQUIRE_FALSE(result->HasError());
    };
    run("SET threads=1");
    run(
      "CREATE TABLE rows AS SELECT i::BIGINT AS id, (i * 0.25)::DOUBLE AS amount, "
      "('label-' || i)::VARCHAR AS label, (i % 100)::SMALLINT AS small, "
      "DATE '2024-01-01' + i::INTEGER AS day, i % 2 = 0 AS flag FROM range(6144) t(i)");
    std::filesystem::create_directories(scratch.path() / "part=2024");
    for (auto const& name : {"a", "missing", "b"}) {
      auto const columns = std::string{name} == "missing" ? "id, amount, label, small, day" : "*";
      run("COPY (SELECT " + std::string{columns} + " FROM rows) TO " +
          scratch.file_literal("part=2024/" + std::string{name} + ".parquet") +
          " (FORMAT PARQUET, ROW_GROUP_SIZE 2048)");
    }
  }

  std::string path(std::string const& name) const
  {
    return scratch.file("part=2024/" + name + ".parquet");
  }

  std::unique_ptr<scan::parquet_ingestible_table_info> make_info(
    std::vector<std::string> const& files,
    std::size_t cap = std::numeric_limits<std::size_t>::max(),
    bool identity   = false) const
  {
    auto info = std::make_unique<scan::parquet_ingestible_table_info>();
    for (auto const& file : files) {
      info->resolved_file_paths.push_back(path(file));
    }
    info->names                  = carrier_names();
    info->returned_types         = carrier_types();
    info->approximate_batch_size = cap;
    if (identity) {
      info->names.pop_back();
      info->returned_types.pop_back();
      for (std::size_t i = 0; i < info->names.size(); ++i) {
        info->column_ids.emplace_back(i);
      }
      info->scan_output_arity = info->names.size();
    } else {
      info->column_ids        = {duckdb::ColumnIndex(duckdb::COLUMN_IDENTIFIER_ROW_ID)};
      info->scan_output_arity = 0;
    }
    return info;
  }

  std::unique_ptr<scan::parquet_ingestible_table_info> make_partition_info(
    std::vector<std::string> const& files,
    sirius::type_id partition_type = sirius::type_id::INTEGER,
    std::size_t cap                = std::numeric_limits<std::size_t>::max()) const
  {
    auto info                  = make_info(files, cap);
    auto const partition_index = info->names.size();
    info->names.push_back("part");
    info->returned_types.push_back(sirius::logical_type::make(partition_type));
    info->column_ids        = {duckdb::ColumnIndex(partition_index)};
    info->partition_indices = {duckdb::HivePartitioningIndex("2024", partition_index)};
    info->scan_output_arity = 1;
    return info;
  }

  std::shared_ptr<scan::parquet_gpu_ingestible> partition_reader(
    std::vector<std::string> const& files,
    sirius::type_id partition_type = sirius::type_id::INTEGER,
    std::size_t cap                = std::numeric_limits<std::size_t>::max()) const
  {
    return scan::make_ingestible(make_partition_info(files, partition_type, cap));
  }

  std::unique_ptr<scan::parquet_file_scan_info> read_file(scan::parquet_gpu_ingestible& reader)
  {
    auto task = reader.next_split_provider(
      [ctx = ioctx](std::string_view) -> std::shared_ptr<sirius::io::sirius_ioctx> { return ctx; });
    REQUIRE(task);
    auto info  = task();
    auto* file = dynamic_cast<scan::parquet_file_scan_info*>(info.get());
    REQUIRE(file);
    REQUIRE(file->row_groups.size() >= 3);
    for (auto const& group : file->row_groups) {
      REQUIRE(group.num_rows > 0);
    }
    info.release();
    return std::unique_ptr<scan::parquet_file_scan_info>(file);
  }
};

std::vector<std::unique_ptr<scan::scan_info>> coalesce_files(
  scan::parquet_gpu_ingestible const& reader,
  std::vector<std::unique_ptr<scan::parquet_file_scan_info>> files)
{
  auto coalescer = reader.create_batch_coalescer();
  std::vector<std::unique_ptr<scan::scan_info>> splits;
  auto append = [&](auto batches) {
    for (auto& batch : batches) {
      splits.push_back(std::move(batch));
    }
  };
  for (auto& file : files) {
    append(coalescer->push(std::move(file)));
  }
  append(coalescer->flush());
  return splits;
}

void check_partition_byte_delta(scan::parquet_file_scan_info const& actual,
                                scan::parquet_file_scan_info const& baseline,
                                std::size_t bytes_per_row)
{
  REQUIRE(actual.file_path == baseline.file_path);
  REQUIRE(actual.row_groups.size() == baseline.row_groups.size());
  for (std::size_t i = 0; i < actual.row_groups.size(); ++i) {
    auto const& group = actual.row_groups[i];
    auto const& base  = baseline.row_groups[i];
    CAPTURE(i, group.num_rows, bytes_per_row);
    REQUIRE(group.index == base.index);
    REQUIRE(group.num_rows == base.num_rows);
    auto const partition_bytes = static_cast<std::size_t>(group.num_rows) * bytes_per_row;
    CHECK(group.output_bytes == base.output_bytes + partition_bytes);
    CHECK(group.decode_working_bytes == base.decode_working_bytes + partition_bytes);
    CHECK(group.compressed_bytes == base.compressed_bytes);
  }
}

}  // namespace

TEST_CASE("parquet scans without a prefetch cache skip advisory ranges",
          "[scan][parquet][prefetch]")
{
  auto ingestible = scan::make_ingestible(make_nation_info(false));
  auto ioctx      = std::make_shared<sirius::io::kvikio_context>();
  REQUIRE_FALSE(ioctx->uses_prefetching_cache());
  auto task = ingestible->next_split_provider(
    [ioctx](std::string_view) -> std::shared_ptr<sirius::io::sirius_ioctx> { return ioctx; });
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
  CHECK(batch->fadvise_entries().empty());
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

  SECTION("virtual-only count star with hive partitions projects a carrier")
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

    CHECK(plan.needs_reader_projection);
    CHECK(plan.is_projected());
    REQUIRE(plan.data_columns.size() == 1);
    CHECK(plan.data_columns.front().primary_idx != 4);
    CHECK(plan.carrier_batch_index == 0);
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

    CHECK(plan.needs_reader_projection);
    CHECK(plan.is_projected());
    REQUIRE(plan.data_columns.size() == 1);
    CHECK(plan.data_columns.front().primary_idx != 4);
    CHECK(plan.carrier_batch_index == 0);
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

TEST_CASE("parquet count star selects one narrow carrier without an output binding",
          "[scan][parquet][scan_plan][carrier]")
{
  auto const names = carrier_names();
  auto const types = carrier_types();
  auto const plan  = scan::build_scan_plan(
    {duckdb::ColumnIndex(duckdb::COLUMN_IDENTIFIER_ROW_ID)}, {}, names, types, 0, {});

  CHECK(plan.output_layout.empty());
  REQUIRE(plan.batch_position_by_column_id.size() == 1);
  check_unmapped_carrier(plan, names, types);
}

TEST_CASE("parquet partition-only scans exclude partition columns from carrier selection",
          "[scan][parquet][hive][scan_plan][carrier]")
{
  auto const names = carrier_names();
  auto const types = carrier_types();
  auto const plan  = scan::build_scan_plan(
    {duckdb::ColumnIndex(5)}, {}, names, types, 1, {duckdb::HivePartitioningIndex("true", 5)});

  REQUIRE(plan.output_layout.size() == 1);
  CHECK(plan.output_layout.front().source == scan::scan_plan::output_entry::PARTITION);
  CHECK(plan.output_layout.front().idx == 0);
  REQUIRE(plan.partition_columns.size() == 1);
  CHECK(plan.partition_columns.front().primary_idx == 5);
  REQUIRE(plan.batch_position_by_column_id.size() == 1);
  check_unmapped_carrier(plan, names, types);
}

TEST_CASE("parquet carrier width ties choose the lowest primary index",
          "[scan][parquet][scan_plan][carrier]")
{
  auto const names = carrier_names();
  auto types       = carrier_types();
  types[3]         = sirius::logical_type::make(sirius::type_id::BOOLEAN);
  REQUIRE(types[3].fixed_width_byte_size() == types[5].fixed_width_byte_size());
  auto const plan = scan::build_scan_plan(
    {duckdb::ColumnIndex(duckdb::COLUMN_IDENTIFIER_ROW_ID)}, {}, names, types, 0, {});

  CHECK(plan.output_layout.empty());
  check_unmapped_carrier(plan, names, types);
  REQUIRE(plan.data_columns.size() == 1);
  CHECK(plan.data_columns.front().primary_idx == 3);
}

TEST_CASE("parquet count star keeps a natural batch without a fixed-width carrier",
          "[scan][parquet][scan_plan][carrier]")
{
  duckdb::vector<std::string> names{"text", "items"};
  duckdb::vector<sirius::logical_type> types{sirius::logical_type::make(sirius::type_id::VARCHAR),
                                             sirius::logical_type::make(sirius::type_id::LIST)};
  auto const plan = scan::build_scan_plan(
    {duckdb::ColumnIndex(duckdb::COLUMN_IDENTIFIER_ROW_ID)}, {}, names, types, 0, {});

  CHECK(plan.data_columns.empty());
  CHECK_FALSE(plan.needs_reader_projection);
  CHECK_FALSE(plan.is_projected());
  CHECK(plan.output_layout.empty());
  REQUIRE(plan.batch_position_by_column_id.size() == 1);
  CHECK_FALSE(plan.batch_position_by_column_id.front().has_value());
}

TEST_CASE("parquet count star without names does not select a carrier",
          "[scan][parquet][scan_plan][carrier]")
{
  auto const plan = scan::build_scan_plan(
    {duckdb::ColumnIndex(duckdb::COLUMN_IDENTIFIER_ROW_ID)}, {}, {}, carrier_types(), 0, {});

  CHECK(plan.data_columns.empty());
  CHECK_FALSE(plan.needs_reader_projection);
  CHECK_FALSE(plan.is_projected());
  CHECK(plan.output_layout.empty());
  REQUIRE(plan.batch_position_by_column_id.size() == 1);
  CHECK_FALSE(plan.batch_position_by_column_id.front().has_value());
}

TEST_CASE("parquet carrier selection leaves nested identity and real projections unchanged",
          "[scan][parquet][scan_plan][carrier]")
{
  duckdb::vector<std::string> names{"flag", "record", "items"};
  duckdb::vector<sirius::logical_type> types{sirius::logical_type::make(sirius::type_id::BOOLEAN),
                                             sirius::logical_type::make(sirius::type_id::STRUCT),
                                             sirius::logical_type::make(sirius::type_id::LIST)};

  SECTION("nested select star retains the identity read")
  {
    auto const plan = scan::build_scan_plan(
      {duckdb::ColumnIndex(0), duckdb::ColumnIndex(1), duckdb::ColumnIndex(2)},
      {},
      names,
      types,
      3,
      {});
    CHECK_FALSE(plan.is_projected());
    CHECK_FALSE(plan.needs_reader_projection);
    REQUIRE(plan.data_columns.size() == 3);
    REQUIRE(plan.output_layout.size() == 3);
    REQUIRE(plan.batch_position_by_column_id.size() == 3);
    for (std::size_t i = 0; i < 3; ++i) {
      CHECK(plan.data_columns[i].primary_idx == i);
      CHECK(plan.output_layout[i].source == scan::scan_plan::output_entry::DATA);
      CHECK(plan.output_layout[i].idx == i);
      REQUIRE(plan.batch_position_by_column_id[i].has_value());
      CHECK(*plan.batch_position_by_column_id[i] == i);
    }
  }

  SECTION("pruned real column does not gain a carrier")
  {
    auto const plan = scan::build_scan_plan({duckdb::ColumnIndex(1)}, {}, names, types, 1, {});
    CHECK(plan.is_projected());
    REQUIRE(plan.data_columns.size() == 1);
    CHECK(plan.data_columns.front().primary_idx == 1);
    REQUIRE(plan.output_layout.size() == 1);
    REQUIRE(plan.batch_position_by_column_id.front().has_value());
    CHECK(*plan.batch_position_by_column_id.front() == 0);
  }

  SECTION("filter-only real column does not gain a narrower carrier")
  {
    auto const plan = scan::build_scan_plan({duckdb::ColumnIndex(1)}, {0}, names, types, 0, {});
    CHECK(plan.is_projected());
    REQUIRE(plan.data_columns.size() == 1);
    CHECK(plan.data_columns.front().primary_idx == 1);
    CHECK(plan.output_layout.empty());
    REQUIRE(plan.batch_position_by_column_id.front().has_value());
    CHECK(*plan.batch_position_by_column_id.front() == 0);
  }
}

TEST_CASE_METHOD(carrier_file_fixture,
                 "parquet carrier fallback reports nonzero bytes for every row group",
                 "[scan][parquet][sizing][carrier]")
{
  auto reader = scan::make_ingestible(make_info({"missing"}));
  auto file   = read_file(*reader);
  CHECK(file->carrier_unavailable);
  REQUIRE(file->reader_options);
  CHECK_FALSE(file->reader_options->get_column_names().has_value());
  for (auto const& group : file->row_groups) {
    CAPTURE(group.index);
    CHECK(group.output_bytes > 0);
    CHECK(group.decode_working_bytes > 0);
    CHECK(group.compressed_bytes > 0);
  }
}

TEST_CASE_METHOD(carrier_file_fixture,
                 "parquet carrier fallback estimates cover the same file's identity read",
                 "[scan][parquet][sizing][carrier]")
{
  auto fallback_reader = scan::make_ingestible(make_info({"missing"}));
  auto identity_reader =
    scan::make_ingestible(make_info({"missing"}, std::numeric_limits<std::size_t>::max(), true));
  auto fallback = read_file(*fallback_reader);
  auto identity = read_file(*identity_reader);
  CHECK(fallback->carrier_unavailable);
  CHECK_FALSE(identity->carrier_unavailable);
  REQUIRE_FALSE(identity->reader_options->get_column_names().has_value());
  REQUIRE(fallback->row_groups.size() == identity->row_groups.size());
  for (std::size_t i = 0; i < fallback->row_groups.size(); ++i) {
    auto const& actual   = fallback->row_groups[i];
    auto const& expected = identity->row_groups[i];
    CAPTURE(i, actual.num_rows);
    REQUIRE(actual.index == expected.index);
    REQUIRE(actual.num_rows == expected.num_rows);
    REQUIRE(expected.output_bytes > 0);
    REQUIRE(expected.decode_working_bytes > 0);
    REQUIRE(expected.compressed_bytes > 0);
    CHECK(actual.output_bytes >= expected.output_bytes);
    CHECK(actual.decode_working_bytes >= expected.decode_working_bytes);
    CHECK(actual.compressed_bytes >= expected.compressed_bytes);
  }
}

TEST_CASE_METHOD(carrier_file_fixture,
                 "parquet carrier fallback respects the coalescer byte cap",
                 "[scan][parquet][sizing][carrier]")
{
  auto probe_reader   = scan::make_ingestible(make_info({"missing"}));
  auto probe          = read_file(*probe_reader);
  auto const smallest = std::min_element(
    probe->row_groups.begin(), probe->row_groups.end(), [](auto const& a, auto const& b) {
      return a.decode_working_bytes < b.decode_working_bytes;
    });
  REQUIRE(smallest->decode_working_bytes > 1);
  auto const cap        = smallest->decode_working_bytes - 1;
  auto fallback_reader  = scan::make_ingestible(make_info({"missing"}, cap));
  auto projected_reader = scan::make_ingestible(make_info({"a"}, cap));
  auto fallback         = read_file(*fallback_reader);
  auto projected        = read_file(*projected_reader);
  CHECK(fallback->carrier_unavailable);
  CHECK_FALSE(projected->carrier_unavailable);
  REQUIRE(fallback->row_groups.size() == projected->row_groups.size());
  auto const group_count = fallback->row_groups.size();
  for (auto const& group : fallback->row_groups) {
    REQUIRE(group.decode_working_bytes > cap);
  }
  std::vector<std::unique_ptr<scan::parquet_file_scan_info>> fallback_files;
  fallback_files.push_back(std::move(fallback));
  auto fallback_splits = coalesce_files(*fallback_reader, std::move(fallback_files));
  std::vector<std::unique_ptr<scan::parquet_file_scan_info>> projected_files;
  projected_files.push_back(std::move(projected));
  auto projected_splits = coalesce_files(*projected_reader, std::move(projected_files));
  CHECK(fallback_splits.size() == group_count);
  CHECK(fallback_splits.size() > 1);
  CHECK(projected_splits.size() < fallback_splits.size());
  for (auto const& info : fallback_splits) {
    auto* split = dynamic_cast<scan::parquet_split_info*>(info.get());
    REQUIRE(split);
    REQUIRE(split->rg_slices.size() == 1);
    CHECK(split->rg_slices.front().row_group_indices.size() == 1);
    CHECK_FALSE(split->reader_options->get_column_names().has_value());
  }
}

TEST_CASE_METHOD(carrier_file_fixture,
                 "parquet carrier fallback sizes physical partition-named columns from metadata",
                 "[scan][parquet][sizing][carrier]")
{
  auto const first_path   = scratch.file("year=2024/a.parquet");
  auto const missing_path = scratch.file("year=2024/missing.parquet");
  {
    sirius::test::scoped_sirius_disable disable;
    duckdb::DuckDB db(nullptr);
    duckdb::Connection con(db);
    auto run = [&](std::string const& sql) {
      auto result = con.Query(sql);
      REQUIRE(result);
      if (result->HasError()) { UNSCOPED_INFO(result->GetError()); }
      REQUIRE_FALSE(result->HasError());
    };
    run("SET threads=1");
    run(
      "CREATE TABLE wide_rows AS SELECT i::BIGINT AS id, "
      "repeat('y', 1024) || i::VARCHAR AS year, i % 2 = 0 AS flag FROM range(6144) t(i)");
    std::filesystem::create_directories(scratch.path() / "year=2024");
    run("COPY wide_rows TO " + scratch.file_literal("year=2024/a.parquet") +
        " (FORMAT PARQUET, ROW_GROUP_SIZE 2048)");
    run("COPY (SELECT id, year FROM wide_rows) TO " +
        scratch.file_literal("year=2024/missing.parquet") +
        " (FORMAT PARQUET, ROW_GROUP_SIZE 2048)");
  }

  auto hive_info                 = std::make_unique<scan::parquet_ingestible_table_info>();
  hive_info->resolved_file_paths = {first_path, missing_path};
  hive_info->names               = {"id", "year", "flag"};
  hive_info->returned_types      = {sirius::logical_type::make(sirius::type_id::BIGINT),
                                    sirius::logical_type::make(sirius::type_id::BIGINT),
                                    sirius::logical_type::make(sirius::type_id::BOOLEAN)};
  hive_info->column_ids          = {duckdb::ColumnIndex(1)};
  hive_info->partition_indices   = {duckdb::HivePartitioningIndex("2024", 1)};
  hive_info->scan_output_arity   = 1;
  auto hive_reader               = scan::make_ingestible(std::move(hive_info));
  auto first                     = read_file(*hive_reader);
  REQUIRE_FALSE(first->carrier_unavailable);
  REQUIRE(first->reader_options->get_column_names().has_value());
  CHECK(*first->reader_options->get_column_names() == std::vector<std::string>{"flag"});
  auto fallback = read_file(*hive_reader);
  REQUIRE(fallback->file_path == missing_path);
  REQUIRE(fallback->carrier_unavailable);
  REQUIRE_FALSE(fallback->reader_options->get_column_names().has_value());
  CHECK(fallback->partition_values == std::vector<std::string>{"2024"});

  auto identity_info                 = std::make_unique<scan::parquet_ingestible_table_info>();
  identity_info->resolved_file_paths = {missing_path};
  identity_info->names               = {"id", "year"};
  identity_info->returned_types      = {sirius::logical_type::make(sirius::type_id::BIGINT),
                                        sirius::logical_type::make(sirius::type_id::VARCHAR)};
  identity_info->column_ids          = {duckdb::ColumnIndex(0), duckdb::ColumnIndex(1)};
  identity_info->scan_output_arity   = 2;
  auto identity_reader               = scan::make_ingestible(std::move(identity_info));
  auto identity                      = read_file(*identity_reader);
  REQUIRE_FALSE(identity->carrier_unavailable);
  REQUIRE_FALSE(identity->reader_options->get_column_names().has_value());
  REQUIRE(identity->partition_values.empty());
  REQUIRE(fallback->row_groups.size() == identity->row_groups.size());
  for (std::size_t i = 0; i < fallback->row_groups.size(); ++i) {
    auto const& actual   = fallback->row_groups[i];
    auto const& expected = identity->row_groups[i];
    CAPTURE(i, actual.num_rows);
    REQUIRE(actual.index == expected.index);
    REQUIRE(actual.num_rows == expected.num_rows);
    REQUIRE(expected.decode_working_bytes >= static_cast<std::size_t>(expected.num_rows) * 512);
    CHECK(actual.decode_working_bytes >= expected.decode_working_bytes);
    CHECK(actual.output_bytes >= expected.output_bytes);
    CHECK(actual.compressed_bytes >= expected.compressed_bytes);
  }
}

TEST_CASE_METHOD(carrier_file_fixture,
                 "parquet integer partition output adds four bytes per row to carrier sizing",
                 "[scan][parquet][sizing][carrier]")
{
  auto reader          = partition_reader({"a"});
  auto baseline_reader = scan::make_ingestible(make_info({"a"}));
  auto file            = read_file(*reader);
  auto baseline        = read_file(*baseline_reader);
  REQUIRE_FALSE(file->carrier_unavailable);
  REQUIRE(file->reader_options->get_column_names().has_value());
  CHECK(*file->reader_options->get_column_names() == std::vector<std::string>{"flag"});
  CHECK(file->partition_values == std::vector<std::string>{"2024"});
  CHECK(baseline->partition_values.empty());
  check_partition_byte_delta(*file, *baseline, 4);
  for (auto const& group : file->row_groups) {
    CHECK(group.decode_working_bytes == group.output_bytes);
  }
}

TEST_CASE_METHOD(
  carrier_file_fixture,
  "parquet varchar partition output includes characters and offsets in carrier sizing",
  "[scan][parquet][sizing][carrier]")
{
  auto reader          = partition_reader({"a"}, sirius::type_id::VARCHAR);
  auto baseline_reader = scan::make_ingestible(make_info({"a"}));
  auto file            = read_file(*reader);
  auto baseline        = read_file(*baseline_reader);
  REQUIRE_FALSE(file->carrier_unavailable);
  CHECK(file->partition_values == std::vector<std::string>{"2024"});
  check_partition_byte_delta(*file, *baseline, 8);
  for (auto const& group : file->row_groups) {
    CHECK(group.decode_working_bytes == group.output_bytes);
  }
}

TEST_CASE_METHOD(carrier_file_fixture,
                 "parquet projected data sizing also includes partition output",
                 "[scan][parquet][sizing][carrier]")
{
  auto info                 = make_partition_info({"a"});
  info->column_ids          = {duckdb::ColumnIndex(0), duckdb::ColumnIndex(info->names.size() - 1)};
  info->scan_output_arity   = 2;
  auto baseline_info        = make_info({"a"});
  baseline_info->column_ids = {duckdb::ColumnIndex(0)};
  baseline_info->scan_output_arity = 1;
  auto reader                      = scan::make_ingestible(std::move(info));
  auto baseline_reader             = scan::make_ingestible(std::move(baseline_info));
  auto file                        = read_file(*reader);
  auto baseline                    = read_file(*baseline_reader);
  REQUIRE_FALSE(file->carrier_unavailable);
  REQUIRE(file->reader_options->get_column_names().has_value());
  REQUIRE(baseline->reader_options->get_column_names().has_value());
  CHECK(*file->reader_options->get_column_names() == std::vector<std::string>{"id"});
  CHECK(*baseline->reader_options->get_column_names() == std::vector<std::string>{"id"});
  CHECK(file->partition_values == std::vector<std::string>{"2024"});
  check_partition_byte_delta(*file, *baseline, 4);
}

TEST_CASE_METHOD(carrier_file_fixture,
                 "parquet partition bytes split carrier batches at the working set cap",
                 "[scan][parquet][sizing][carrier]")
{
  auto probe_reader             = scan::make_ingestible(make_info({"a", "b"}));
  std::size_t carrier_bytes     = 0;
  std::size_t partitioned_bytes = 0;
  std::size_t cap               = 0;
  for (auto const& name : {"a", "b"}) {
    auto file = read_file(*probe_reader);
    REQUIRE(file->file_path == path(name));
    REQUIRE_FALSE(file->carrier_unavailable);
    std::size_t file_bytes = 0;
    for (auto const& group : file->row_groups) {
      carrier_bytes += group.decode_working_bytes;
      file_bytes += group.decode_working_bytes + static_cast<std::size_t>(group.num_rows) * 4;
    }
    partitioned_bytes += file_bytes;
    cap = std::max(cap, file_bytes);
  }
  // Fit one whole partitioned file, rather than splitting each of its row groups.
  REQUIRE(carrier_bytes <= cap);
  REQUIRE(cap < partitioned_bytes);
  auto reader          = partition_reader({"a", "b"}, sirius::type_id::INTEGER, cap);
  auto baseline_reader = scan::make_ingestible(make_info({"a", "b"}, cap));
  std::vector<std::unique_ptr<scan::parquet_file_scan_info>> files;
  std::vector<std::unique_ptr<scan::parquet_file_scan_info>> baseline_files;
  std::vector<std::size_t> group_counts;
  for (auto const& name : {"a", "b"}) {
    auto file = read_file(*reader);
    REQUIRE(file->file_path == path(name));
    group_counts.push_back(file->row_groups.size());
    files.push_back(std::move(file));
    baseline_files.push_back(read_file(*baseline_reader));
  }
  auto baseline_splits = coalesce_files(*baseline_reader, std::move(baseline_files));
  REQUIRE(baseline_splits.size() == 1);
  CHECK(baseline_splits.front()->estimated_working_set_bytes() == carrier_bytes);
  auto splits = coalesce_files(*reader, std::move(files));
  REQUIRE(splits.size() == 2);
  std::vector<std::string> const names{"a", "b"};
  std::size_t total_working = 0;
  for (std::size_t i = 0; i < splits.size(); ++i) {
    auto* split = dynamic_cast<scan::parquet_split_info*>(splits[i].get());
    REQUIRE(split);
    REQUIRE(split->rg_slices.size() == 1);
    CHECK(split->rg_slices.front().file_path == path(names[i]));
    CHECK(split->rg_slices.front().row_group_indices.size() == group_counts[i]);
    CHECK(split->estimated_working_set_bytes() <= cap);
    total_working += split->estimated_working_set_bytes();
  }
  CHECK(total_working == partitioned_bytes);
}

TEST_CASE_METHOD(carrier_file_fixture,
                 "parquet carrier fallback sizing includes partition output",
                 "[scan][parquet][sizing][carrier]")
{
  auto reader = partition_reader({"missing"});
  auto identity_reader =
    scan::make_ingestible(make_info({"missing"}, std::numeric_limits<std::size_t>::max(), true));
  auto file     = read_file(*reader);
  auto identity = read_file(*identity_reader);
  REQUIRE(file->carrier_unavailable);
  REQUIRE_FALSE(file->reader_options->get_column_names().has_value());
  REQUIRE_FALSE(identity->reader_options->get_column_names().has_value());
  CHECK(file->partition_values == std::vector<std::string>{"2024"});
  REQUIRE(file->row_groups.size() == identity->row_groups.size());
  for (std::size_t i = 0; i < file->row_groups.size(); ++i) {
    auto const& actual   = file->row_groups[i];
    auto const& expected = identity->row_groups[i];
    CAPTURE(i, actual.num_rows);
    REQUIRE(actual.index == expected.index);
    REQUIRE(actual.num_rows == expected.num_rows);
    auto const partition_bytes = static_cast<std::size_t>(actual.num_rows) * 4;
    CHECK(actual.output_bytes >= expected.output_bytes + partition_bytes);
    CHECK(actual.decode_working_bytes >= expected.decode_working_bytes + partition_bytes);
    CHECK(actual.compressed_bytes == expected.compressed_bytes);
  }
}

TEST_CASE_METHOD(carrier_file_fixture,
                 "parquet count star sizing does not charge unprojected hive partitions",
                 "[scan][parquet][sizing][carrier]")
{
  auto info               = make_partition_info({"a"});
  info->column_ids        = {duckdb::ColumnIndex(duckdb::COLUMN_IDENTIFIER_ROW_ID)};
  info->scan_output_arity = 0;
  auto reader             = scan::make_ingestible(std::move(info));
  auto baseline_reader    = scan::make_ingestible(make_info({"a"}));
  auto file               = read_file(*reader);
  auto baseline           = read_file(*baseline_reader);
  REQUIRE_FALSE(file->carrier_unavailable);
  CHECK(file->partition_values.empty());
  check_partition_byte_delta(*file, *baseline, 0);
  std::vector<std::unique_ptr<scan::parquet_file_scan_info>> files;
  files.push_back(std::move(file));
  auto splits = coalesce_files(*reader, std::move(files));
  REQUIRE(splits.size() == 1);
  auto* split = dynamic_cast<scan::parquet_split_info*>(splits.front().get());
  REQUIRE(split);
  REQUIRE(split->plan);
  CHECK_FALSE(split->plan->has_partitions());
  CHECK(split->plan->partition_primary_indices.count(carrier_names().size()) == 1);
  CHECK(split->plan->output_layout.empty());
  CHECK(split->plan->carrier_batch_index.has_value());
}

TEST_CASE_METHOD(carrier_file_fixture,
                 "parquet coalescer separates carrier and natural options within one partition",
                 "[scan][parquet][carrier][coalesce]")
{
  auto reader = partition_reader({"a", "missing", "b"});
  std::vector<std::unique_ptr<scan::parquet_file_scan_info>> files;
  for (auto const& name : {"a", "missing", "b"}) {
    auto file = read_file(*reader);
    CHECK(file->file_path == path(name));
    REQUIRE(file->partition_values == std::vector<std::string>{"2024"});
    REQUIRE_FALSE(file->disable_filter_pushdown);
    files.push_back(std::move(file));
  }
  CHECK_FALSE(files[0]->carrier_unavailable);
  CHECK(files[1]->carrier_unavailable);
  CHECK_FALSE(files[2]->carrier_unavailable);
  auto const projected_options = files[0]->reader_options;
  auto const natural_options   = files[1]->reader_options;
  REQUIRE(projected_options == files[2]->reader_options);
  REQUIRE(projected_options != natural_options);
  auto splits = coalesce_files(*reader, std::move(files));
  REQUIRE(splits.size() == 3);
  std::vector<std::string> const names{"a", "missing", "b"};
  for (std::size_t i = 0; i < splits.size(); ++i) {
    auto* split = dynamic_cast<scan::parquet_split_info*>(splits[i].get());
    REQUIRE(split);
    REQUIRE(split->rg_slices.size() == 1);
    CHECK(split->rg_slices.front().file_path == path(names[i]));
    CHECK(split->reader_options == (i == 1 ? natural_options : projected_options));
    CHECK(split->reader_options->get_column_names().has_value() == (i != 1));
    CHECK(split->partition_values == std::vector<std::string>{"2024"});
  }
}

TEST_CASE_METHOD(carrier_file_fixture,
                 "parquet coalescer combines adjacent files with the same carrier options",
                 "[scan][parquet][carrier][coalesce]")
{
  auto reader = partition_reader({"a", "b"});
  std::vector<std::unique_ptr<scan::parquet_file_scan_info>> files;
  for (auto const& name : {"a", "b"}) {
    auto file = read_file(*reader);
    CHECK(file->file_path == path(name));
    REQUIRE_FALSE(file->carrier_unavailable);
    REQUIRE(file->partition_values == std::vector<std::string>{"2024"});
    REQUIRE_FALSE(file->disable_filter_pushdown);
    files.push_back(std::move(file));
  }
  auto const options = files[0]->reader_options;
  REQUIRE(options == files[1]->reader_options);
  auto splits = coalesce_files(*reader, std::move(files));
  REQUIRE(splits.size() == 1);
  auto* split = dynamic_cast<scan::parquet_split_info*>(splits.front().get());
  REQUIRE(split);
  REQUIRE(split->rg_slices.size() == 2);
  CHECK(split->rg_slices[0].file_path == path("a"));
  CHECK(split->rg_slices[1].file_path == path("b"));
  CHECK(split->reader_options == options);
  CHECK(split->reader_options->get_column_names().has_value());
}
