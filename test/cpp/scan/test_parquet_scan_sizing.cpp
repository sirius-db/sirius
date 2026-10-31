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

#include <cudf/strings/utilities.hpp>

#include <catch.hpp>
#include <cucascade/exec/semi_future.hpp>
#include <duckdb.hpp>
#include <duckdb/common/constants.hpp>
#include <duckdb/common/multi_file/multi_file_reader.hpp>
#include <duckdb/execution/column_binding_resolver.hpp>
#include <duckdb/parser/parser.hpp>
#include <duckdb/planner/expression/bound_cast_expression.hpp>
#include <duckdb/planner/filter/conjunction_filter.hpp>
#include <duckdb/planner/filter/constant_filter.hpp>
#include <duckdb/planner/filter/in_filter.hpp>
#include <duckdb/planner/logical_operator.hpp>
#include <duckdb/planner/planner.hpp>
#include <op/dynamic_filter/sirius_dynamic_filter.hpp>
#include <op/scan/parquet_gpu_ingestible.hpp>
#include <op/scan/parquet_schema_mapping.hpp>
#include <op/scan/parquet_source_filter.hpp>
#include <op/scan/scan_plan.hpp>
#include <op/scan/sirius_gpu_scan_operator_data.hpp>
#include <operator/operator_test_utils.hpp>
#include <utils/local_ioctx.hpp>
#include <utils/parquet_fixture_utils.hpp>
#include <utils/table_filter_test_utils.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
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
    info->projection_ids = {duckdb::ProjectionIndex(0)};
  } else {
    info->column_ids.push_back(duckdb::ColumnIndex(0));
  }
  if (pure_filter && !zero_output) {
    info->column_ids.push_back(duckdb::ColumnIndex(3));
    info->projection_ids = {duckdb::ProjectionIndex(0), duckdb::ProjectionIndex(1)};
  }
  info->projected_output_count = zero_output ? 0 : 1;
  info->approximate_batch_size = std::size_t{1} << 30;
  return info;
}

std::unique_ptr<scan::parquet_ingestible_table_info> make_partition_only_nation_info()
{
  auto info               = make_nation_info(true);
  info->column_ids        = {duckdb::ColumnIndex(2), duckdb::ColumnIndex(3)};
  info->projection_ids    = {duckdb::ProjectionIndex(0), duckdb::ProjectionIndex(1)};
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
  auto ioctx      = sirius::test::make_test_local_ioctx();
  auto task       = ingestible->next_split_provider(
    [ioctx](std::string_view) -> std::shared_ptr<cucascade::io::ioctx> { return ioctx; });
  REQUIRE(task);

  auto file = task({});
  REQUIRE(file);
  auto const file_output  = file->estimated_bytes();
  auto const file_working = file->estimated_working_set_bytes();
  CHECK(file->preparation_bytes() == file_working);

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
  CHECK(batch->preparation_bytes() == file_working);
  return {batch->estimated_bytes(),
          batch->estimated_working_set_bytes(),
          batch->plan->non_output_batch_positions().size()};
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
  std::shared_ptr<cucascade::io::ioctx> ioctx = sirius::test::make_test_local_ioctx();

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
    run("COPY (SELECT * FROM rows) TO " + scratch.file_literal("part=2024/路径-非常长.parquet") +
        " (FORMAT PARQUET, ROW_GROUP_SIZE 2048)");
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
      info->projected_output_count = info->names.size();
    } else {
      info->column_ids             = {duckdb::ColumnIndex(duckdb::COLUMN_IDENTIFIER_ROW_ID)};
      info->projected_output_count = 0;
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
    info->column_ids             = {duckdb::ColumnIndex(partition_index)};
    info->partition_indices      = {duckdb::HivePartitioningIndex("2024", partition_index)};
    info->projected_output_count = 1;
    return info;
  }

  std::unique_ptr<scan::parquet_ingestible_table_info> make_virtual_info(
    std::vector<std::string> const& files, bool reader_mode = false) const
  {
    auto info                    = make_info(files);
    auto const id                = duckdb::MultiFileReader::COLUMN_IDENTIFIER_FILE_ROW_NUMBER;
    info->column_ids             = {duckdb::ColumnIndex(id)};
    info->projected_output_count = 1;
    info->virtual_columns        = {
      {id, "file_row_number", sirius::logical_type::make(sirius::type_id::BIGINT), std::nullopt}};
    if (reader_mode) {
      info->sirius_dynamic_filters = std::make_shared<sirius::op::sirius_dynamic_filter_set>();
    }
    return info;
  }

  std::shared_ptr<scan::parquet_gpu_ingestible> partition_reader(
    std::vector<std::string> const& files,
    sirius::type_id partition_type = sirius::type_id::INTEGER,
    std::size_t cap                = std::numeric_limits<std::size_t>::max()) const
  {
    return scan::make_ingestible(make_partition_info(files, partition_type, cap));
  }

  std::unique_ptr<scan::parquet_file_scan_info> read_file(scan::parquet_gpu_ingestible& reader,
                                                          std::size_t minimum_groups = 3)
  {
    auto task = reader.next_split_provider(
      [ctx = ioctx](std::string_view) -> std::shared_ptr<cucascade::io::ioctx> { return ctx; });
    REQUIRE(task);
    auto info  = task({});
    auto* file = dynamic_cast<scan::parquet_file_scan_info*>(info.get());
    REQUIRE(file);
    REQUIRE(file->row_groups.size() >= minimum_groups);
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

TEST_CASE("parquet scans without a prefetch cache retain advisory ranges",
          "[scan][parquet][prefetch]")
{
  auto ingestible = scan::make_ingestible(make_nation_info(false));
  auto ioctx      = sirius::test::make_test_local_ioctx();
  REQUIRE_FALSE(ioctx->uses_fs_cache());
  auto task = ingestible->next_split_provider(
    [ioctx](std::string_view) -> std::shared_ptr<cucascade::io::ioctx> { return ioctx; });
  REQUIRE(task);

  auto coalescer = ingestible->create_batch_coalescer();
  auto batches   = coalescer->push(task({}));
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
  info->contract_id            = 1;
  scan::parquet_gpu_ingestible ingestible{std::move(info)};
  auto coalescer = ingestible.create_batch_coalescer();

  auto file = std::make_unique<scan::parquet_file_scan_info>();
  file->set_contract_payload(
    1, {{.contract_id = 1, .split_id = 0, .input_identity = "sizing.parquet"}}, {{}});
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

TEST_CASE("parquet virtual batches apply only their source mode allocation model",
          "[scan][parquet][sizing][virtual_columns]")
{
  auto const kind =
    GENERATE(scan::parquet_source_kind::PLAIN_PARQUET, scan::parquet_source_kind::ICEBERG);
  bool const legacy = kind == scan::parquet_source_kind::ICEBERG;
  auto make_reader  = [kind](std::size_t cap) {
    auto info            = std::make_unique<scan::parquet_ingestible_table_info>();
    info->names          = {"x"};
    info->returned_types = {sirius::logical_type::make(sirius::type_id::INTEGER)};
    info->column_ids = {duckdb::ColumnIndex(duckdb::MultiFileReader::COLUMN_IDENTIFIER_FILENAME)};
    info->projected_output_count = 1;
    info->approximate_batch_size = cap;
    info->contract_id            = 1;
    info->virtual_columns        = {{duckdb::MultiFileReader::COLUMN_IDENTIFIER_FILENAME,
                                     "filename",
                                     sirius::logical_type::make(sirius::type_id::VARCHAR),
                                     scan::scan_plan::parquet_virtual_column_kind::FILENAME}};
    return std::make_shared<scan::parquet_gpu_ingestible>(std::move(info), kind);
  };
  auto make_file = [] {
    auto file = std::make_unique<scan::parquet_file_scan_info>();
    file->set_contract_payload(
      1, {{.contract_id = 1, .split_id = 0, .input_identity = "sizing.parquet"}}, {{}});
    file->row_groups.push_back({0, 20, 60, 10, 1});
    file->row_groups.push_back({1, 20, 60, 10, 1});
    return file;
  };

  SECTION("reservation includes inputs and concatenated result")
  {
    auto reader    = make_reader(std::numeric_limits<std::size_t>::max());
    auto coalescer = reader->create_batch_coalescer();
    CHECK(coalescer->retained_bytes() == 0);
    CHECK(coalescer->push(make_file()).empty());
    CHECK(coalescer->retained_bytes() == (legacy ? 240 : 120));
    auto splits = coalescer->flush();
    CHECK(coalescer->retained_bytes() == 0);
    REQUIRE(splits.size() == 1);
    CHECK(splits.front()->estimated_working_set_bytes() == (legacy ? 240 : 120));
  }

  SECTION("coalescer applies the concatenation peak to its byte cap")
  {
    auto reader    = make_reader(200);
    auto coalescer = reader->create_batch_coalescer();
    auto splits    = coalescer->push(make_file());
    auto tail      = coalescer->flush();
    for (auto& split : tail) {
      splits.push_back(std::move(split));
    }
    REQUIRE(splits.size() == (legacy ? 2 : 1));
    for (auto const& split : splits) {
      CHECK(split->estimated_working_set_bytes() == (legacy ? 60 : 120));
    }
  }

  SECTION("separate files share the peak budget and flush resets the run count")
  {
    auto reader    = make_reader(240);
    auto coalescer = reader->create_batch_coalescer();
    auto push_one  = [&] {
      auto file = make_file();
      file->row_groups.resize(1);
      return coalescer->push(std::move(file));
    };
    CHECK(push_one().empty());
    CHECK(coalescer->retained_bytes() == 60);
    CHECK(push_one().empty());
    CHECK(coalescer->retained_bytes() == (legacy ? 240 : 120));
    auto full = push_one();
    REQUIRE(full.size() == (legacy ? 1 : 0));
    if (legacy) {
      CHECK(full.front()->estimated_working_set_bytes() == 240);
      CHECK(full.front()->estimated_bytes() == 40);
    }
    auto tail = coalescer->flush();
    REQUIRE(tail.size() == 1);
    CHECK(tail.front()->estimated_working_set_bytes() == (legacy ? 60 : 180));
    CHECK(tail.front()->estimated_bytes() == (legacy ? 20 : 60));
    auto const* split = dynamic_cast<scan::parquet_split_info*>(tail.front().get());
    REQUIRE(split);
    CHECK(split->source_kind == kind);
  }
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

TEST_CASE("parquet scans report the output prefix their bound plan reads",
          "[scan][parquet][assembly]")
{
  SECTION("a trailing pure-filter column lies past the prefix")
  {
    auto const ingestible = scan::make_ingestible(make_nation_info(true));
    REQUIRE(ingestible->materialized_column_order().size() == 2);
    CHECK(ingestible->output_prefix_width() == std::size_t{1});
    CHECK(ingestible->output_assembly_is_leading_identity());
  }

  SECTION("an output-only scan is its own prefix")
  {
    auto const ingestible = scan::make_ingestible(make_nation_info(false));
    REQUIRE(ingestible->materialized_column_order().size() == 1);
    CHECK(ingestible->output_prefix_width() == std::size_t{1});
    CHECK(ingestible->output_assembly_is_leading_identity());
  }

  SECTION("a zero-output scan reads no column")
  {
    auto const ingestible = scan::make_ingestible(make_nation_info(true, true));
    CHECK_FALSE(ingestible->output_prefix_width().has_value());
    CHECK_FALSE(ingestible->output_assembly_is_leading_identity());
  }

  SECTION("a partition-only scan reads no column")
  {
    auto const ingestible = scan::make_ingestible(make_partition_only_nation_info());
    CHECK_FALSE(ingestible->output_prefix_width().has_value());
    CHECK_FALSE(ingestible->output_assembly_is_leading_identity());
  }
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
                                      /*projected_output_count=*/0,
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
                                      /*projected_output_count=*/0,
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
                                      /*projected_output_count=*/1,
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
                                      /*projected_output_count=*/4,
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

  auto hive_info                    = std::make_unique<scan::parquet_ingestible_table_info>();
  hive_info->resolved_file_paths    = {first_path, missing_path};
  hive_info->names                  = {"id", "year", "flag"};
  hive_info->returned_types         = {sirius::logical_type::make(sirius::type_id::BIGINT),
                                       sirius::logical_type::make(sirius::type_id::BIGINT),
                                       sirius::logical_type::make(sirius::type_id::BOOLEAN)};
  hive_info->column_ids             = {duckdb::ColumnIndex(1)};
  hive_info->partition_indices      = {duckdb::HivePartitioningIndex("2024", 1)};
  hive_info->projected_output_count = 1;
  hive_info->sirius_dynamic_filters = std::make_shared<sirius::op::sirius_dynamic_filter_set>();
  auto hive_reader                  = scan::make_ingestible(std::move(hive_info));
  auto first                        = read_file(*hive_reader);
  REQUIRE_FALSE(first->carrier_unavailable);
  REQUIRE(first->reader_options->get_column_names().has_value());
  CHECK(*first->reader_options->get_column_names() == std::vector<std::string>{"flag"});
  auto fallback = read_file(*hive_reader);
  REQUIRE(fallback->file_path == missing_path);
  REQUIRE(fallback->carrier_unavailable);
  REQUIRE_FALSE(fallback->reader_options->get_column_names().has_value());
  CHECK(fallback->partition_values == std::vector<std::string>{"2024"});

  auto identity_info                    = std::make_unique<scan::parquet_ingestible_table_info>();
  identity_info->resolved_file_paths    = {missing_path};
  identity_info->names                  = {"id", "year"};
  identity_info->returned_types         = {sirius::logical_type::make(sirius::type_id::BIGINT),
                                           sirius::logical_type::make(sirius::type_id::VARCHAR)};
  identity_info->column_ids             = {duckdb::ColumnIndex(0), duckdb::ColumnIndex(1)};
  identity_info->projected_output_count = 2;
  auto identity_reader                  = scan::make_ingestible(std::move(identity_info));
  auto identity                         = read_file(*identity_reader);
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
  auto info        = make_partition_info({"a"});
  info->column_ids = {duckdb::ColumnIndex(0), duckdb::ColumnIndex(info->names.size() - 1)};
  info->projected_output_count          = 2;
  auto baseline_info                    = make_info({"a"});
  baseline_info->column_ids             = {duckdb::ColumnIndex(0)};
  baseline_info->projected_output_count = 1;
  auto reader                           = scan::make_ingestible(std::move(info));
  auto baseline_reader                  = scan::make_ingestible(std::move(baseline_info));
  auto file                             = read_file(*reader);
  auto baseline                         = read_file(*baseline_reader);
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
  auto reader = scan::make_ingestible(make_partition_info({"missing"}));
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
  auto info                    = make_partition_info({"a"});
  info->column_ids             = {duckdb::ColumnIndex(duckdb::COLUMN_IDENTIFIER_ROW_ID)};
  info->projected_output_count = 0;
  auto reader                  = scan::make_ingestible(std::move(info));
  auto baseline_reader         = scan::make_ingestible(make_info({"a"}));
  auto file                    = read_file(*reader);
  auto baseline                = read_file(*baseline_reader);
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
                 "parquet filename virtual sizing charges UTF-8 paths offsets and scratch",
                 "[scan][parquet][sizing][virtual_columns]")
{
  // A one-byte cap forces one row group per split: no concatenation allowance can hide
  // missing filename-construction scratch in the admission estimate.
  auto info        = make_info({"路径-非常长"}, 1);
  info->column_ids = {duckdb::ColumnIndex(duckdb::MultiFileReader::COLUMN_IDENTIFIER_FILENAME)};
  info->projected_output_count = 1;
  info->virtual_columns        = {{duckdb::MultiFileReader::COLUMN_IDENTIFIER_FILENAME,
                                   "filename",
                                   sirius::logical_type::make(sirius::type_id::VARCHAR),
                                   scan::scan_plan::parquet_virtual_column_kind::FILENAME}};
  // Compare physical-reader estimates; metadata-only synthesis has no decoded
  // carrier validity mask and therefore a different output footprint.
  info->sirius_dynamic_filters = std::make_shared<sirius::op::sirius_dynamic_filter_set>();
  auto reader                  = scan::make_ingestible(std::move(info));
  auto baseline_reader         = scan::make_ingestible(make_info({"路径-非常长"}));
  auto file                    = read_file(*reader);
  auto baseline                = read_file(*baseline_reader);
  REQUIRE(file->row_groups.size() == baseline->row_groups.size());
  for (std::size_t i = 0; i < file->row_groups.size(); ++i) {
    auto const rows  = static_cast<std::size_t>(file->row_groups[i].num_rows);
    auto const chars = rows * file->file_path.size();
    auto const offset_width =
      cudf::strings::is_large_strings_enabled() &&
          chars >= static_cast<std::size_t>(cudf::strings::get_offset64_threshold())
        ? 8
        : 4;
    auto const filename_bytes = chars + (rows + 1) * offset_width;
    CHECK(file->row_groups[i].output_bytes ==
          baseline->row_groups[i].output_bytes + filename_bytes);
    CHECK(file->row_groups[i].decode_working_bytes >= baseline->row_groups[i].decode_working_bytes +
                                                        filename_bytes + rows * 16 +
                                                        file->file_path.size());
  }
  auto const groups = file->row_groups;
  std::vector<std::unique_ptr<scan::parquet_file_scan_info>> files;
  files.push_back(std::move(file));
  auto splits = coalesce_files(*reader, std::move(files));
  REQUIRE(splits.size() == groups.size());
  for (std::size_t i = 0; i < splits.size(); ++i) {
    CHECK(splits[i]->estimated_working_set_bytes() == groups[i].decode_working_bytes);
  }
}

TEST_CASE("parquet filename sizing handles offset thresholds and overflow",
          "[scan][parquet][sizing][virtual_columns]")
{
  auto const threshold = static_cast<std::size_t>(cudf::strings::get_offset64_threshold());
  auto const wide      = cudf::strings::is_large_strings_enabled();
  SECTION("offset width changes at the threshold, including the terminal offset")
  {
    auto const below = scan::estimate_filename_column_size(1, threshold - 1);
    CHECK(below.output_bytes == threshold - 1 + 2 * 4);
    CHECK(below.working_bytes == below.output_bytes + 16 + threshold - 1);
    for (auto const chars : {threshold, threshold + 1}) {
      auto const at_or_above = scan::estimate_filename_column_size(1, chars);
      CHECK(at_or_above.output_bytes == chars + 2 * (wide ? 8 : 4));
      CHECK(at_or_above.working_bytes == at_or_above.output_bytes + 16 + chars);
    }
  }
  SECTION("typed empty output needs no scalar buffer")
  {
    auto const empty = scan::estimate_filename_column_size(0, threshold);
    CHECK(empty.output_bytes == 0);
    CHECK(empty.working_bytes == 0);
  }
  SECTION("individual products and sums saturate rather than wrapping")
  {
    auto const max = std::numeric_limits<std::size_t>::max();
    // Character multiplication, terminal offset count, and character/offset addition.
    for (auto const [rows, path_bytes] : {std::pair{std::size_t{2}, max},
                                          std::pair{max, std::size_t{0}},
                                          std::pair{std::size_t{1}, max - 4}}) {
      auto const size = scan::estimate_filename_column_size(rows, path_bytes);
      CHECK(size.output_bytes == max);
      CHECK(size.working_bytes == max);
    }
    // Pair multiplication and output-plus-pairs addition can overflow independently.
    for (auto const rows : {max / 16 + 1, max / 20 + 1}) {
      auto const size = scan::estimate_filename_column_size(rows, 0);
      CHECK(size.output_bytes < max);
      CHECK(size.working_bytes == max);
    }
  }
}

TEST_CASE_METHOD(carrier_file_fixture,
                 "parquet all-pruned completion preserves empty virtual column types",
                 "[scan][parquet][sizing][virtual_columns]")
{
  auto info        = make_info({"路径-非常长"});
  info->column_ids = {duckdb::ColumnIndex(duckdb::MultiFileReader::COLUMN_IDENTIFIER_FILENAME)};
  info->projected_output_count = 1;
  info->virtual_columns        = {{duckdb::MultiFileReader::COLUMN_IDENTIFIER_FILENAME,
                                   "filename",
                                   sirius::logical_type::make(sirius::type_id::VARCHAR),
                                   scan::scan_plan::parquet_virtual_column_kind::FILENAME}};
  if (GENERATE(false, true)) {
    info->sirius_dynamic_filters = std::make_shared<sirius::op::sirius_dynamic_filter_set>();
  }
  auto reader = scan::make_ingestible(std::move(info));
  auto file   = read_file(*reader);
  // Feed the coalescer the same shape as a fully stats-pruned metadata task.
  file->row_groups.clear();
  std::vector<std::unique_ptr<scan::parquet_file_scan_info>> files;
  files.push_back(std::move(file));
  auto splits = coalesce_files(*reader, std::move(files));
  REQUIRE(splits.size() == 1);
  auto const* split = dynamic_cast<scan::parquet_split_info*>(splits.front().get());
  REQUIRE(split);
  REQUIRE(split->rg_slices.size() == 1);
  CHECK(split->rg_slices.front().row_group_indices.empty());
  CHECK(split->estimated_bytes() == 0);
  CHECK(split->estimated_working_set_bytes() == 0);
  CHECK(split->rg_slices.front().reserved_compressed_bytes == 0);
  CHECK(split->fadvise_hints().empty());
  auto manager      = sirius::test::operator_utils::initialize_memory_manager();
  auto const* space = manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
  REQUIRE(space);
  auto decoded = reader->materialize_metadata_to_table(
    *split, *space, cudf::get_default_stream(), false, nullptr);
  auto output = reader->post_filter_and_project(
    std::move(decoded), *space, cudf::get_default_stream(), false, nullptr, nullptr, {});
  REQUIRE(output->num_columns() == 1);
  CHECK(output->num_rows() == 0);
  CHECK(output->view().column(0).type().id() == cudf::type_id::STRING);
}

TEST_CASE_METHOD(carrier_file_fixture,
                 "parquet coalescer separates carrier and natural options within one partition",
                 "[scan][parquet][carrier][coalesce]")
{
  auto reader = scan::make_ingestible(make_partition_info({"a", "missing", "b"}));
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

TEST_CASE("parquet source filters distinguish unsupported from SQL NULL",
          "[scan][parquet][source_pruning]")
{
  using result       = scan::source_filter_result;
  auto const value   = duckdb::Value::UBIGINT(7);
  auto current_equal = sirius::test::constant_filter(duckdb::ExpressionType::COMPARE_EQUAL, value);
  CHECK(scan::evaluate_parquet_source_filter(*current_equal, value) == result::MATCH);
  CHECK(scan::evaluate_parquet_source_filter(*current_equal, duckdb::Value::UBIGINT(9)) ==
        result::NO_MATCH);
  auto current_null = sirius::test::null_filter(duckdb::LogicalType::UBIGINT);
  CHECK(scan::evaluate_parquet_source_filter(*current_null, value) == result::NO_MATCH);
  duckdb::LegacyConstantFilter equal(duckdb::ExpressionType::COMPARE_EQUAL,
                                     duckdb::Value::UBIGINT(7));
  CHECK(scan::evaluate_parquet_source_filter(equal, value) == result::MATCH);
  CHECK(scan::evaluate_parquet_source_filter(equal, duckdb::Value::UBIGINT(9)) == result::NO_MATCH);
  duckdb::LegacyConstantFilter mismatch(duckdb::ExpressionType::COMPARE_EQUAL, duckdb::Value("7"));
  CHECK(scan::evaluate_parquet_source_filter(mismatch, value) == result::UNKNOWN);
  CHECK(scan::evaluate_parquet_source_filter(equal, duckdb::Value(duckdb::LogicalType::UBIGINT)) ==
        result::SQL_NULL);
  duckdb::LegacyConstantFilter different(duckdb::ExpressionType::COMPARE_EQUAL,
                                         duckdb::Value::UBIGINT(3));
  duckdb::LegacyConjunctionOrFilter either;
  either.child_filters.push_back(std::make_unique<duckdb::LegacyConstantFilter>(
    duckdb::ExpressionType::COMPARE_EQUAL, duckdb::Value("7")));
  either.child_filters.push_back(std::make_unique<duckdb::LegacyConstantFilter>(
    duckdb::ExpressionType::COMPARE_EQUAL, duckdb::Value::UBIGINT(3)));
  CHECK(scan::evaluate_parquet_source_filter(either, value) == result::UNKNOWN);
  either.child_filters.push_back(std::make_unique<duckdb::LegacyConstantFilter>(
    duckdb::ExpressionType::COMPARE_EQUAL, duckdb::Value::UBIGINT(7)));
  CHECK(scan::evaluate_parquet_source_filter(either, value) == result::MATCH);
  duckdb::LegacyInFilter in({duckdb::Value::UBIGINT(3), duckdb::Value::UBIGINT(5)});
  CHECK(scan::evaluate_parquet_source_filter(in, duckdb::Value(duckdb::LogicalType::UBIGINT)) ==
        result::SQL_NULL);
  CHECK(scan::evaluate_parquet_source_filter(in, duckdb::Value::UBIGINT(3)) == result::MATCH);
}

TEST_CASE("parquet source pruning evaluates current expression filters conservatively",
          "[scan][parquet][source_pruning]")
{
  using result     = scan::source_filter_result;
  using type       = duckdb::ExpressionType;
  auto const value = duckdb::Value::UBIGINT(7);
  auto const null  = duckdb::Value(duckdb::LogicalType::UBIGINT);
  auto in          = sirius::test::in_filter({duckdb::Value::UBIGINT(3), value, null});
  CHECK(scan::evaluate_parquet_source_filter(*in, value) == result::MATCH);
  CHECK(scan::evaluate_parquet_source_filter(*in, duckdb::Value::UBIGINT(9)) == result::SQL_NULL);
  CHECK(scan::evaluate_parquet_source_filter(*in, null) == result::SQL_NULL);
  auto comparison_null = sirius::test::constant_filter(type::COMPARE_EQUAL, null);
  CHECK(scan::evaluate_parquet_source_filter(*comparison_null, value) == result::SQL_NULL);
  auto comparison = sirius::test::constant_filter(type::COMPARE_EQUAL, value);
  CHECK(scan::evaluate_parquet_source_filter(*comparison, null) == result::SQL_NULL);
  auto distinct_null = sirius::test::constant_filter(type::COMPARE_DISTINCT_FROM, null);
  CHECK(scan::evaluate_parquet_source_filter(*distinct_null, value) == result::UNKNOWN);
  auto between = sirius::test::conjunction_filter(
    type::CONJUNCTION_AND,
    sirius::test::constant_filter(type::COMPARE_GREATERTHAN, duckdb::Value::UBIGINT(3)),
    sirius::test::constant_filter(type::COMPARE_LESSTHAN, duckdb::Value::UBIGINT(9)));
  CHECK(scan::evaluate_parquet_source_filter(*between, value) == result::MATCH);
  CHECK(scan::evaluate_parquet_source_filter(*between, duckdb::Value::UBIGINT(10)) ==
        result::NO_MATCH);
  auto either = sirius::test::conjunction_filter(
    type::CONJUNCTION_OR,
    sirius::test::constant_filter(type::COMPARE_EQUAL, duckdb::Value::UBIGINT(3)),
    sirius::test::dynamic_filter(duckdb::LogicalType::UBIGINT));
  CHECK(scan::evaluate_parquet_source_filter(*either, value) == result::UNKNOWN);
  CHECK(scan::evaluate_parquet_source_filter(*either, duckdb::Value::UBIGINT(3)) == result::MATCH);
  auto filename =
    sirius::test::constant_filter(type::COMPARE_EQUAL, duckdb::Value("s3://bucket/a"));
  CHECK(scan::evaluate_parquet_source_filter(*filename, duckdb::Value("s3://bucket/a")) ==
        result::MATCH);
  CHECK(scan::evaluate_parquet_source_filter(*filename, duckdb::Value("s3://bucket/b")) ==
        result::NO_MATCH);
}

TEST_CASE("parquet source conjunctions preserve SQL NULL and unsupported branches",
          "[scan][parquet][source_pruning]")
{
  using result     = scan::source_filter_result;
  using type       = duckdb::ExpressionType;
  constexpr auto m = result::MATCH, n = result::NO_MATCH, s = result::SQL_NULL, u = result::UNKNOWN;
  // Rows/columns are TRUE, FALSE, SQL NULL, unsupported. UNKNOWN must keep all
  // possible TRUE outcomes; a known FALSE conjunct can still rule a file out.
  constexpr std::array<std::array<result, 4>, 4> and_results{
    {{{m, n, s, u}}, {{n, n, n, n}}, {{s, n, s, u}}, {{u, n, u, u}}}};
  constexpr std::array<std::array<result, 4>, 4> or_results{
    {{{m, m, m, m}}, {{m, n, s, u}}, {{m, s, s, u}}, {{m, u, u, u}}}};
  auto const legacy = GENERATE(false, true);
  auto const is_and = GENERATE(false, true);
  auto const left   = GENERATE(0, 1, 2, 3);
  auto const right  = GENERATE(0, 1, 2, 3);
  CAPTURE(legacy, is_and, left, right);
  auto operand = [&](int state) -> duckdb::unique_ptr<duckdb::TableFilter> {
    auto value = state == 0   ? duckdb::Value::UBIGINT(7)
                 : state == 1 ? duckdb::Value::UBIGINT(9)
                 : state == 2 ? duckdb::Value(duckdb::LogicalType::UBIGINT)
                              : duckdb::Value("7");
    if (legacy) {
      if (state == 2) {
        // Legacy constant/IN constructors forbid NULL constants. The evaluator still
        // accepts ExpressionFilter children inside a legacy conjunction.
        return sirius::test::constant_filter(type::COMPARE_EQUAL, value);
      }
      return duckdb::make_uniq<duckdb::LegacyConstantFilter>(type::COMPARE_EQUAL, value);
    }
    if (state == 3) { return sirius::test::dynamic_filter(duckdb::LogicalType::UBIGINT); }
    return sirius::test::constant_filter(type::COMPARE_EQUAL, value);
  };
  duckdb::unique_ptr<duckdb::TableFilter> filter;
  if (legacy) {
    if (is_and) {
      auto conjunction = duckdb::make_uniq<duckdb::LegacyConjunctionAndFilter>();
      conjunction->child_filters.push_back(operand(left));
      conjunction->child_filters.push_back(operand(right));
      filter = std::move(conjunction);
    } else {
      auto conjunction = duckdb::make_uniq<duckdb::LegacyConjunctionOrFilter>();
      conjunction->child_filters.push_back(operand(left));
      conjunction->child_filters.push_back(operand(right));
      filter = std::move(conjunction);
    }
  } else {
    filter = sirius::test::conjunction_filter(
      is_and ? type::CONJUNCTION_AND : type::CONJUNCTION_OR, operand(left), operand(right));
  }
  CHECK(scan::evaluate_parquet_source_filter(*filter, duckdb::Value::UBIGINT(7)) ==
        (is_and ? and_results : or_results)[left][right]);
}

TEST_CASE("parquet source pruning keeps unsupported expression shapes",
          "[scan][parquet][source_pruning]")
{
  using result = scan::source_filter_result;
  using type   = duckdb::ExpressionType;
  auto ref     = [] {
    return duckdb::make_uniq<duckdb::BoundReferenceExpression>(duckdb::LogicalType::UBIGINT, 0);
  };
  auto constant = [](duckdb::Value value) {
    return duckdb::make_uniq<duckdb::BoundConstantExpression>(std::move(value));
  };
  auto evaluate = [&](duckdb::unique_ptr<duckdb::Expression> expression) {
    duckdb::ExpressionFilter filter(std::move(expression));
    return scan::evaluate_parquet_source_filter(filter, duckdb::Value::UBIGINT(7));
  };
  CHECK(evaluate(duckdb::BoundComparisonExpression::Create(
          type::COMPARE_LESSTHAN, constant(duckdb::Value::UBIGINT(9)), ref())) == result::UNKNOWN);
  for (bool with_null : {false, true}) {
    auto not_in = duckdb::make_uniq<duckdb::BoundOperatorExpression>(type::COMPARE_NOT_IN,
                                                                     duckdb::LogicalType::BOOLEAN);
    not_in->GetChildrenMutable().push_back(ref());
    not_in->GetChildrenMutable().push_back(constant(duckdb::Value::UBIGINT(9)));
    if (with_null) {
      not_in->GetChildrenMutable().push_back(constant(duckdb::Value(duckdb::LogicalType::UBIGINT)));
    }
    CHECK(evaluate(std::move(not_in)) == result::UNKNOWN);
  }
  auto cast =
    duckdb::BoundCastExpression::AddDefaultCastToType(ref(), duckdb::LogicalType::VARCHAR);
  CHECK(evaluate(duckdb::BoundComparisonExpression::Create(
          type::COMPARE_EQUAL, std::move(cast), constant(duckdb::Value("7")))) == result::UNKNOWN);
  auto filename = sirius::test::constant_filter(type::COMPARE_EQUAL, duckdb::Value("A.parquet"));
  CHECK(scan::evaluate_parquet_source_filter(*filename, duckdb::Value("a.parquet")) ==
        result::NO_MATCH);
  CHECK(scan::evaluate_parquet_source_filter(*filename, duckdb::Value("A.parquet")) ==
        result::MATCH);
}

TEST_CASE("parquet source pruning retains DuckDB bound collation expressions",
          "[scan][parquet][source_pruning]")
{
  sirius::test::scoped_sirius_disable disable;
  duckdb::DuckDB db(nullptr);
  duckdb::Connection con(db);
  REQUIRE_FALSE(con.Query("BEGIN TRANSACTION")->HasError());
  for (auto const* predicate :
       {"filename COLLATE nocase = 'a.parquet'",
        "filename COLLATE nocase = 'a.parquet' OR filename = 'B.parquet'"}) {
    CAPTURE(predicate);
    // Bind real SQL without optimization so constant folding cannot remove the filter.
    duckdb::Parser parser(*con.context);
    parser.ParseQuery("SELECT filename FROM (VALUES ('A.parquet')) input(filename) WHERE " +
                      std::string{predicate});
    duckdb::Planner planner(*con.context);
    planner.CreatePlan(std::move(parser.statements[0]));
    planner.plan->ResolveOperatorTypes();
    duckdb::ColumnBindingResolver resolver;
    resolver.VisitOperator(*planner.plan);
    auto* node = planner.plan.get();
    while (node->type != duckdb::LogicalOperatorType::LOGICAL_FILTER &&
           node->children.size() == 1) {
      node = node->children[0].get();
    }
    REQUIRE(node->type == duckdb::LogicalOperatorType::LOGICAL_FILTER);
    REQUIRE(node->expressions.size() == 1);
    duckdb::ExpressionFilter filter(node->expressions[0]->Copy());
    auto const value = duckdb::Value("A.parquet");
    CHECK(filter.EvaluateWithConstant(*con.context, value));
    CHECK(scan::evaluate_parquet_source_filter(filter, value) ==
          scan::source_filter_result::UNKNOWN);
  }
  REQUIRE_FALSE(con.Query("ROLLBACK")->HasError());
}

TEST_CASE_METHOD(carrier_file_fixture,
                 "parquet virtual scans coalesce equivalent replacement carrier projections",
                 "[scan][parquet][carrier][provenance_mechanism]")
{
  std::filesystem::copy_file(path("missing"), path("missing2"));
  auto reader = scan::make_ingestible(make_virtual_info({"missing", "missing2"}, true));
  std::vector<std::unique_ptr<scan::parquet_file_scan_info>> files;
  files.push_back(read_file(*reader));
  files.push_back(read_file(*reader));
  REQUIRE(files[0]->reader_options != files[1]->reader_options);
  REQUIRE(files[0]->reader_options->get_column_names() ==
          std::optional<std::vector<std::string>>{{"small"}});
  auto splits = coalesce_files(*reader, std::move(files));
  REQUIRE(splits.size() == 1);
  auto const& split = dynamic_cast<scan::parquet_split_info const&>(*splits.front());
  REQUIRE(split.rg_slices.size() == 2);
  auto manager      = sirius::test::operator_utils::initialize_memory_manager();
  auto const* space = manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
  REQUIRE(space);
  auto decoded = reader->materialize_metadata_to_table(
    split, *space, cudf::get_default_stream(), false, nullptr);
  CHECK(decoded.table.num_rows() == 2 * 6144);
  CHECK(reader->reader_calls() == 1);
}

TEST_CASE_METHOD(carrier_file_fixture,
                 "parquet virtual reader retains supported bind carriers outside narrow selection",
                 "[scan][parquet][carrier][provenance_mechanism]")
{
  auto const decimal = scratch.file("decimal.parquet");
  {
    sirius::test::scoped_sirius_disable disable;
    duckdb::DuckDB db(nullptr);
    duckdb::Connection con(db);
    auto result = con.Query("COPY (SELECT range::DECIMAL(18,2) AS d FROM range(7)) TO " +
                            sirius::test::sql_literal(decimal) + " (FORMAT PARQUET)");
    REQUIRE_FALSE(result->HasError());
  }
  auto info                 = make_virtual_info({}, true);
  info->resolved_file_paths = {decimal};
  info->names               = {"d"};
  info->returned_types      = {sirius::logical_type::make_decimal(18, 2)};
  auto reader               = scan::make_ingestible(std::move(info));
  auto file                 = read_file(*reader, 1);
  REQUIRE(file->reader_options->get_column_names() ==
          std::optional<std::vector<std::string>>{{"d"}});
  std::vector<std::unique_ptr<scan::parquet_file_scan_info>> files;
  files.push_back(std::move(file));
  auto splits = coalesce_files(*reader, std::move(files));
  REQUIRE(splits.size() == 1);
  auto manager      = sirius::test::operator_utils::initialize_memory_manager();
  auto const* space = manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
  REQUIRE(space);
  auto decoded = reader->materialize_metadata_to_table(
    *splits.front(), *space, cudf::get_default_stream(), false, nullptr);
  auto output = reader->post_filter_and_project(
    std::move(decoded), *space, cudf::get_default_stream(), false, nullptr, nullptr, {});
  REQUIRE(output->num_columns() == 1);
  REQUIRE(output->num_rows() == 7);
  std::vector<int64_t> positions(7);
  REQUIRE(cudaMemcpy(positions.data(),
                     output->view().column(0).data<int64_t>(),
                     positions.size() * sizeof(int64_t),
                     cudaMemcpyDeviceToHost) == cudaSuccess);
  CHECK(positions == std::vector<int64_t>{0, 1, 2, 3, 4, 5, 6});
  CHECK(reader->reader_calls() == 1);
}

TEST_CASE_METHOD(
  carrier_file_fixture,
  "parquet metadata-only validates footer evidence without decoding physical columns",
  "[scan][parquet][metadata_only][profile]")
{
  auto const strip_evidence = GENERATE(false, true);
  auto const virtual_output = GENERATE(false, true);
  auto info                 = virtual_output ? make_virtual_info({"a"}) : make_info({"a"});
  info->injections.synthetic_parquet_codec   = "LZO";
  info->injections.strip_encryption_evidence = strip_evidence;
  auto reader                                = scan::make_ingestible(std::move(info));
  if (!virtual_output || strip_evidence) {
    REQUIRE_THROWS_AS(read_file(*reader), scan::unsupported_physical_input);
    return;
  }
  auto file = read_file(*reader);
  REQUIRE(file->certificates().size() == 1);
  auto const& certificate = file->certificates().front();
  CHECK(certificate.validation.test(static_cast<unsigned>(scan::later_check::footer_per_file)));
  CHECK(certificate.validation.test(static_cast<unsigned>(scan::later_check::profile_per_file)));
  CHECK(file->dependencies().front().profiles->get(certificate.profile).columns.empty());
  std::vector<std::unique_ptr<scan::parquet_file_scan_info>> files;
  files.push_back(std::move(file));
  auto splits = coalesce_files(*reader, std::move(files));
  REQUIRE(splits.size() == 1);
  CHECK(splits.front()->fadvise_hints().empty());
  auto manager      = sirius::test::operator_utils::initialize_memory_manager();
  auto const* space = manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
  REQUIRE(space);
  auto decoded = reader->materialize_metadata_to_table(
    *splits.front(), *space, cudf::get_default_stream(), false, nullptr);
  CHECK(decoded.table.num_rows() == 6144);
  CHECK(reader->reader_calls() == 0);
}

TEST_CASE_METHOD(carrier_file_fixture,
                 "parquet count retains its physical reader and carrier type",
                 "[scan][parquet][carrier][provenance_mechanism]")
{
  auto info = make_info({"a"});
  if (GENERATE(false, true)) {
    info->sirius_dynamic_filters = std::make_shared<sirius::op::sirius_dynamic_filter_set>();
  }
  auto reader = scan::make_ingestible(std::move(info));
  std::vector<std::unique_ptr<scan::parquet_file_scan_info>> files;
  files.push_back(read_file(*reader));
  auto splits = coalesce_files(*reader, std::move(files));
  REQUIRE(splits.size() == 1);
  CHECK_FALSE(splits.front()->fadvise_hints().empty());
  auto manager      = sirius::test::operator_utils::initialize_memory_manager();
  auto const* space = manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
  REQUIRE(space);
  auto decoded = reader->materialize_metadata_to_table(
    *splits.front(), *space, cudf::get_default_stream(), false, nullptr);
  CHECK(decoded.table.num_rows() == 6144);
  CHECK(decoded.table.view().column(0).type().id() == cudf::type_id::BOOL8);
  CHECK(reader->reader_calls() == 1);
}

TEST_CASE_METHOD(carrier_file_fixture,
                 "parquet source pruning resolves only surviving original file identities",
                 "[scan][parquet][source_pruning]")
{
  struct counted_ioctx : cucascade::io::ioctx {
    size_t opens = 0;
    ~counted_ioctx() override { pre_destroy(); }
    cucascade::io::io_context_type type() const noexcept override
    {
      return cucascade::io::io_context_type::uring;
    }
    void shutdown() noexcept override {}
    bool supports(std::string_view) const noexcept override { return true; }
    bool supports_device_read() const noexcept override { return false; }
    bool supports_host_to_device_read() const noexcept override { return false; }
    bool supports_vector_host_read() const noexcept override { return false; }
    bool supports_device_range_read() const noexcept override { return false; }
    std::vector<cucascade::io::byte_range> align_and_coalesce(
      std::span<cucascade::io::byte_range const> ranges,
      std::optional<size_t>) const noexcept override
    {
      return {ranges.begin(), ranges.end()};
    }
    size_t host_read_io(cucascade::io::io_object const&, size_t, size_t, uint8_t*) override
    {
      throw std::runtime_error("pruned file must not be read");
    }
    cucascade::exec::semi_future<size_t> mixed_readv_async_io(
      cucascade::io::io_object const&,
      std::vector<cucascade::io::prepared_io_slice>&&) noexcept override
    {
      return cucascade::exec::make_semi_future<size_t>(
        std::make_exception_ptr(std::runtime_error("pruned file must not be read")));
    }
    std::shared_ptr<cucascade::io::io_object> create_io_object(std::string) override
    {
      ++opens;
      throw std::runtime_error("pruned file must not be opened");
    }
  };
  auto observed_io           = std::make_shared<counted_ioctx>();
  auto const select_none     = GENERATE(false, true);
  auto const resolver_throws = GENERATE(false, true);
  auto info                  = make_info({"a", "missing", "b"});
  auto counters              = std::make_shared<scan::physical_check_counters>();
  counters->track_units      = true;
  size_t footers = 0, parsed = 0;
  counters->parquet_phase_for_testing    = [&](auto const&, bool footer) { footers += footer; };
  counters->parquet_metadata_for_testing = [&](auto const&, bool) { ++parsed; };
  info->profiles->counters               = counters;
  auto const id                          = duckdb::MultiFileReader::COLUMN_IDENTIFIER_FILE_INDEX;
  info->column_ids                       = {duckdb::ColumnIndex(id)};
  info->projected_output_count           = 1;
  // Native IDs have no explicit kind: the canonical plan resolves their identity.
  info->virtual_columns = {
    {id, "file_index", sirius::logical_type::make(sirius::type_id::UBIGINT), std::nullopt}};
  info->table_filters = duckdb::make_uniq<duckdb::TableFilterSet>();
  info->table_filters->PushFilter(
    duckdb::ProjectionIndex(0),
    sirius::test::constant_filter(duckdb::ExpressionType::COMPARE_EQUAL,
                                  duckdb::Value::UBIGINT(select_none ? 99 : 2)));
  auto reader = scan::make_ingestible(std::move(info));
  std::vector<std::string> resolved;
  auto task = reader->next_split_provider(
    [&](std::string_view name) -> std::shared_ptr<cucascade::io::ioctx> {
      resolved.emplace_back(name);
      if (select_none && resolver_throws) {
        throw std::runtime_error("pruned file must not be resolved");
      }
      if (select_none) { return observed_io; }
      return ioctx;
    });
  if (select_none) {
    CHECK(reader->source_selection_empty());
    CHECK(reader->has_processed_all_metadata());
    CHECK_FALSE(task);
    CHECK(resolved.empty());
    CHECK(observed_io->opens == 0);
    CHECK(footers == 0);
    CHECK(parsed == 0);
    CHECK(reader->create_batch_coalescer()->flush().empty());
    CHECK(reader->reader_calls() == 0);
    return;
  }
  REQUIRE(task);
  auto metadata    = task({});
  auto const* file = dynamic_cast<scan::parquet_file_scan_info*>(metadata.get());
  REQUIRE(file);
  CHECK(file->file_index == 2);
  REQUIRE(resolved.size() == 1);
  CHECK(resolved.front() == path("b"));
  CHECK(footers == 1);
  CHECK(parsed == 1);
  CHECK(reader->has_processed_all_metadata());
  CHECK_FALSE(reader->next_split_provider([&](std::string_view) { return ioctx; }));
  CHECK_FALSE(file->row_groups.empty());
}

TEST_CASE_METHOD(carrier_file_fixture,
                 "parquet metadata-only splits budget synthesis and request no data prefetch",
                 "[scan][parquet][metadata_only]")
{
  auto info                    = make_info({"a", "missing", "b"});
  auto const id                = duckdb::MultiFileReader::COLUMN_IDENTIFIER_FILE_ROW_NUMBER;
  info->column_ids             = {duckdb::ColumnIndex(id)};
  info->projected_output_count = 1;
  info->virtual_columns        = {
    {id, "file_row_number", sirius::logical_type::make(sirius::type_id::BIGINT), std::nullopt}};
  auto reader      = scan::make_ingestible(std::move(info));
  auto coalescer   = reader->create_batch_coalescer();
  std::size_t rows = 0;
  std::vector<std::unique_ptr<scan::scan_info>> splits;
  while (!reader->has_processed_all_metadata()) {
    auto file = read_file(*reader);
    CHECK(file->fadvise_hints().empty());
    for (auto const& rg : file->row_groups) {
      rows += rg.num_rows;
      CHECK(rg.compressed_bytes == 0);
      CHECK(rg.decode_working_bytes >= static_cast<std::size_t>(rg.num_rows));
    }
    auto emitted = coalescer->push(std::move(file));
    for (auto& split : emitted) {
      splits.push_back(std::move(split));
    }
  }
  auto tail = coalescer->flush();
  for (auto& split : tail) {
    splits.push_back(std::move(split));
  }
  CHECK(rows == 3 * 6144);
  REQUIRE_FALSE(splits.empty());
  for (auto const& split : splits) {
    CHECK(split->fadvise_hints().empty());
    CHECK(split->estimated_working_set_bytes() > 0);
    CHECK(dynamic_cast<scan::parquet_split_info const&>(*split).read_mode ==
          scan::parquet_read_mode::METADATA_ONLY);
  }
}

TEST_CASE_METHOD(carrier_file_fixture,
                 "parquet reader calls follow split mode rather than row-group count",
                 "[scan][parquet][provenance_mechanism]")
{
  bool const metadata_only = GENERATE(false, true);
  auto info                = make_info({"a", "b"});
  auto const row_id        = duckdb::MultiFileReader::COLUMN_IDENTIFIER_FILE_ROW_NUMBER;
  info->column_ids         = {duckdb::ColumnIndex(row_id)};
  info->virtual_columns    = {
    {row_id, "file_row_number", sirius::logical_type::make(sirius::type_id::BIGINT), std::nullopt}};
  info->projected_output_count = 1;
  if (!metadata_only) {
    info->sirius_dynamic_filters = std::make_shared<sirius::op::sirius_dynamic_filter_set>();
  }
  auto reader = scan::make_ingestible(std::move(info));
  std::vector<std::unique_ptr<scan::parquet_file_scan_info>> files;
  std::size_t groups = 0;
  while (!reader->has_processed_all_metadata()) {
    auto file = read_file(*reader);
    groups += file->row_groups.size();
    files.push_back(std::move(file));
  }
  auto splits = coalesce_files(*reader, std::move(files));
  REQUIRE(splits.size() == 1);
  REQUIRE(groups == 6);
  auto manager      = sirius::test::operator_utils::initialize_memory_manager();
  auto const* space = manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
  REQUIRE(space);
  std::size_t rows = 0;
  for (auto const& split : splits) {
    auto decoded = reader->materialize_metadata_to_table(
      *split, *space, cudf::get_default_stream(), false, nullptr);
    auto table = reader->post_filter_and_project(
      std::move(decoded), *space, cudf::get_default_stream(), false, nullptr, nullptr, {});
    CHECK(table->num_columns() == 1);
    rows += table->num_rows();
    std::vector<int64_t> row_numbers(table->num_rows());
    REQUIRE(cudaMemcpy(row_numbers.data(),
                       table->view().column(0).data<int64_t>(),
                       row_numbers.size() * sizeof(int64_t),
                       cudaMemcpyDeviceToHost) == cudaSuccess);
    for (std::size_t i = 0; i < row_numbers.size(); ++i) {
      CHECK(row_numbers[i] == static_cast<int64_t>(i % 6144));
    }
    if (metadata_only) { CHECK(split->fadvise_hints().empty()); }
    auto const* resource = space->get_memory_resource_of<cucascade::memory::Tier::GPU>();
    REQUIRE(resource);
    auto const peak = resource->get_peak_total_allocated_bytes();
    CAPTURE(metadata_only, peak, split->estimated_working_set_bytes());
    CHECK(peak > 0);
    CHECK(peak <= split->estimated_working_set_bytes());
  }
  CHECK(rows == 2 * 6144);
  CHECK(reader->reader_calls() == (metadata_only ? 0 : splits.size()));
}

TEST_CASE_METHOD(carrier_file_fixture,
                 "parquet virtual carrier projects a narrow column present in each file",
                 "[scan][parquet][carrier][provenance_mechanism]")
{
  auto info             = make_info({"a", "missing", "b"});
  auto const row_id     = duckdb::MultiFileReader::COLUMN_IDENTIFIER_FILE_ROW_NUMBER;
  info->column_ids      = {duckdb::ColumnIndex(row_id)};
  info->virtual_columns = {
    {row_id, "file_row_number", sirius::logical_type::make(sirius::type_id::BIGINT), std::nullopt}};
  info->projected_output_count = 1;
  info->sirius_dynamic_filters = std::make_shared<sirius::op::sirius_dynamic_filter_set>();
  auto reader                  = scan::make_ingestible(std::move(info));
  auto first                   = read_file(*reader);
  auto middle                  = read_file(*reader);
  auto last                    = read_file(*reader);
  REQUIRE(first->reader_options->get_column_names());
  REQUIRE(middle->reader_options->get_column_names());
  REQUIRE(last->reader_options->get_column_names());
  CHECK(*first->reader_options->get_column_names() == std::vector<std::string>{"flag"});
  CHECK(*middle->reader_options->get_column_names() == std::vector<std::string>{"small"});
  CHECK(*last->reader_options->get_column_names() == std::vector<std::string>{"flag"});
  CHECK_FALSE(middle->carrier_unavailable);
  std::vector<std::unique_ptr<scan::parquet_file_scan_info>> files;
  files.push_back(std::move(first));
  files.push_back(std::move(middle));
  files.push_back(std::move(last));
  auto splits = coalesce_files(*reader, std::move(files));
  CHECK(splits.size() == 3);
}

TEST_CASE_METHOD(carrier_file_fixture,
                 "parquet provenance reader applies late physical dynamic filters before residuals",
                 "[scan][parquet][provenance_mechanism][scan_merge]")
{
  bool const partial_static    = GENERATE(false, true);
  auto info                    = make_info({"a", "b"});
  info->sirius_dynamic_filters = std::make_shared<sirius::op::sirius_dynamic_filter_set>();
  auto filters                 = info->sirius_dynamic_filters;
  auto const row_id            = duckdb::MultiFileReader::COLUMN_IDENTIFIER_FILE_ROW_NUMBER;
  auto const file_id           = duckdb::MultiFileReader::COLUMN_IDENTIFIER_FILE_INDEX;
  info->column_ids             = {
    duckdb::ColumnIndex(row_id), duckdb::ColumnIndex(file_id), duckdb::ColumnIndex(0)};
  info->virtual_columns = {
    {row_id, "file_row_number", sirius::logical_type::make(sirius::type_id::BIGINT), std::nullopt},
    {file_id, "file_index", sirius::logical_type::make(sirius::type_id::UBIGINT), std::nullopt}};
  info->projected_output_count = 3;
  if (partial_static) {
    info->table_filters = duckdb::make_uniq<duckdb::TableFilterSet>();
    info->table_filters->PushFilter(
      duckdb::ProjectionIndex(2),
      sirius::test::conjunction_filter(
        duckdb::ExpressionType::CONJUNCTION_AND,
        sirius::test::constant_filter(duckdb::ExpressionType::COMPARE_GREATERTHANOREQUALTO,
                                      duckdb::Value::BIGINT(2048)),
        sirius::test::constant_filter(duckdb::ExpressionType::COMPARE_NOTEQUAL,
                                      duckdb::Value::BIGINT(3000))));
    info->table_filters->PushFilter(
      duckdb::ProjectionIndex(0),
      sirius::test::constant_filter(duckdb::ExpressionType::COMPARE_GREATERTHANOREQUALTO,
                                    duckdb::Value::BIGINT(3002)));
  }
  auto reader = scan::make_ingestible(std::move(info));
  std::vector<std::unique_ptr<scan::parquet_file_scan_info>> files;
  while (!reader->has_processed_all_metadata()) {
    auto file = read_file(*reader, partial_static ? 2 : 3);
    CHECK(file->row_groups.size() == (partial_static ? 2 : 3));
    files.push_back(std::move(file));
  }
  auto splits = coalesce_files(*reader, std::move(files));
  REQUIRE(splits.size() == 1);
  auto manager      = sirius::test::operator_utils::initialize_memory_manager();
  auto const* space = manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
  REQUIRE(space);
  // Publish only after the metadata tasks completed. O index 2 is physical id;
  // O indexes 0/1 are M-space virtual slots and must never index D directly.
  std::vector<sirius::op::zone_map_entry> zones;
  zones.push_back({std::make_unique<cudf::numeric_scalar<int64_t>>(3000),
                   std::make_unique<cudf::numeric_scalar<int64_t>>(3002)});
  auto producer = filters->register_producer({2});
  REQUIRE(producer.push_filter(
    2, std::make_shared<sirius::op::sirius_dynamic_zone_map_filter>(std::move(zones))));
  auto decoded = reader->materialize_metadata_to_table(
    *splits.front(), *space, cudf::get_default_stream(), false, nullptr);
  CHECK(decoded.table.view().num_rows() == (partial_static ? 4 : 6));
  CHECK(decoded.state == scan::filter_state::UNFILTERED);
  CHECK(reader->reader_calls() == 1);
  auto output = reader->post_filter_and_project(
    std::move(decoded), *space, cudf::get_default_stream(), false, nullptr, nullptr, {});
  REQUIRE(output->num_columns() == 3);
  REQUIRE(output->num_rows() == (partial_static ? 2 : 6));
  std::vector<int64_t> rows(output->num_rows());
  std::vector<int64_t> ids(output->num_rows());
  std::vector<uint64_t> sources(output->num_rows());
  REQUIRE(cudaMemcpy(rows.data(),
                     output->view().column(0).data<int64_t>(),
                     rows.size() * sizeof(int64_t),
                     cudaMemcpyDeviceToHost) == cudaSuccess);
  REQUIRE(cudaMemcpy(sources.data(),
                     output->view().column(1).data<uint64_t>(),
                     sources.size() * sizeof(uint64_t),
                     cudaMemcpyDeviceToHost) == cudaSuccess);
  REQUIRE(cudaMemcpy(ids.data(),
                     output->view().column(2).data<int64_t>(),
                     ids.size() * sizeof(int64_t),
                     cudaMemcpyDeviceToHost) == cudaSuccess);
  for (std::size_t i = 0; i < rows.size(); ++i) {
    CHECK(rows[i] == ids[i]);
    CHECK(rows[i] >= (partial_static ? 3002 : 3000));
    CHECK(rows[i] <= 3002);
    CHECK(sources[i] == i / (partial_static ? 1 : 3));
  }
  // Release published scalar owners before their memory manager.
  filters.reset();
  reader.reset();
}

TEST_CASE_METHOD(carrier_file_fixture,
                 "parquet source pruning composes with reader and residual row filters",
                 "[scan][parquet][source_pruning][provenance_mechanism]")
{
  using type           = duckdb::ExpressionType;
  auto const selection = GENERATE(0, 1, 2, 3);  // file_index, filename, no files, empty last file
  auto const small_splits = GENERATE(false, true);
  CAPTURE(selection, small_splits);
  auto const last = selection == 3 ? "high" : "b";
  if (selection == 3) {
    sirius::test::scoped_sirius_disable disable;
    duckdb::DuckDB db(nullptr);
    duckdb::Connection writer(db);
    auto result = writer.Query(
      "COPY (SELECT id + 100000 AS id, amount, label, small, day, flag FROM read_parquet(" +
      sirius::test::sql_literal(path("b")) + ")) TO " + sirius::test::sql_literal(path(last)) +
      " (FORMAT PARQUET, ROW_GROUP_SIZE 2048)");
    REQUIRE(result);
    REQUIRE_FALSE(result->HasError());
  }
  auto info                    = make_info({"a", "missing", last},
                        small_splits ? 1024 : std::numeric_limits<std::size_t>::max());
  auto const row_id            = duckdb::MultiFileReader::COLUMN_IDENTIFIER_FILE_ROW_NUMBER;
  auto const file_id           = duckdb::MultiFileReader::COLUMN_IDENTIFIER_FILE_INDEX;
  auto const name_id           = duckdb::MultiFileReader::COLUMN_IDENTIFIER_FILENAME;
  info->column_ids             = {duckdb::ColumnIndex(row_id),
                                  duckdb::ColumnIndex(file_id),
                                  duckdb::ColumnIndex(0),
                                  duckdb::ColumnIndex(name_id)};
  info->projection_ids         = {duckdb::ProjectionIndex(0),
                                  duckdb::ProjectionIndex(1),
                                  duckdb::ProjectionIndex(2),
                                  duckdb::ProjectionIndex(3)};
  info->projected_output_count = 3;
  info->virtual_columns        = {
    {row_id, "file_row_number", sirius::logical_type::make(sirius::type_id::BIGINT), std::nullopt},
    {file_id, "file_index", sirius::logical_type::make(sirius::type_id::UBIGINT), std::nullopt},
    {name_id, "filename", sirius::logical_type::make(sirius::type_id::VARCHAR), std::nullopt}};
  info->table_filters = duckdb::make_uniq<duckdb::TableFilterSet>();
  if (selection == 1) {
    info->table_filters->PushFilter(
      duckdb::ProjectionIndex(3),
      sirius::test::in_filter({duckdb::Value(path("a")), duckdb::Value(path(last))}));
  } else {
    info->table_filters->PushFilter(
      duckdb::ProjectionIndex(1),
      sirius::test::in_filter(
        selection == 2
          ? duckdb::vector<duckdb::Value>{duckdb::Value::UBIGINT(99)}
          : duckdb::vector<duckdb::Value>{duckdb::Value::UBIGINT(0), duckdb::Value::UBIGINT(2)}));
  }
  auto groups = sirius::test::conjunction_filter(
    type::CONJUNCTION_OR,
    sirius::test::constant_filter(type::COMPARE_LESSTHAN, duckdb::Value::BIGINT(2048)),
    sirius::test::constant_filter(type::COMPARE_GREATERTHANOREQUALTO, duckdb::Value::BIGINT(4096)));
  info->table_filters->PushFilter(
    duckdb::ProjectionIndex(2),
    sirius::test::conjunction_filter(
      type::CONJUNCTION_AND,
      std::move(groups),
      sirius::test::constant_filter(type::COMPARE_LESSTHAN, duckdb::Value::BIGINT(6143))));
  info->table_filters->PushFilter(
    duckdb::ProjectionIndex(0),
    sirius::test::constant_filter(type::COMPARE_GREATERTHANOREQUALTO, duckdb::Value::BIGINT(1000)));
  auto reader = scan::make_ingestible(std::move(info));
  std::vector<std::string> opened;
  auto coalescer = reader->create_batch_coalescer();
  std::vector<std::unique_ptr<scan::scan_info>> splits;
  while (auto task = reader->next_split_provider([&](std::string_view name) {
    opened.emplace_back(name);
    if (selection == 2) { throw std::runtime_error("all files were eliminated"); }
    return ioctx;
  })) {
    auto file            = task({});
    auto const& metadata = dynamic_cast<scan::parquet_file_scan_info const&>(*file);
    CHECK(metadata.file_index != 1);
    if (!metadata.row_groups.empty()) {
      REQUIRE(metadata.row_groups.size() == 2);
      CHECK(metadata.row_groups[0].index == 0);
      CHECK(metadata.row_groups[1].index == 2);
    }
    for (auto& split : coalescer->push(std::move(file))) {
      splits.push_back(std::move(split));
    }
  }
  for (auto& split : coalescer->flush()) {
    splits.push_back(std::move(split));
  }
  CHECK(opened == (selection == 2 ? std::vector<std::string>{}
                                  : std::vector<std::string>{path("a"), path(last)}));
  if (selection == 2) {
    CHECK(splits.empty());
    CHECK(reader->has_processed_all_metadata());
    CHECK(reader->reader_calls() == 0);
    return;
  }
  REQUIRE_FALSE(splits.empty());
  if (selection != 2) { CHECK(splits.size() == (small_splits ? (selection == 3 ? 2 : 4) : 1)); }
  auto manager      = sirius::test::operator_utils::initialize_memory_manager();
  auto const* space = manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
  REQUIRE(space);
  std::vector<bool> seen(3 * 6144, false);
  std::size_t decoded_rows = 0, output_rows = 0;
  for (auto const& split : splits) {
    auto decoded = reader->materialize_metadata_to_table(
      *split, *space, cudf::get_default_stream(), false, nullptr);
    decoded_rows += decoded.table.num_rows();
    if (decoded.table.num_rows()) { CHECK(decoded.state == scan::filter_state::UNFILTERED); }
    auto output = reader->post_filter_and_project(
      std::move(decoded), *space, cudf::get_default_stream(), false, nullptr, nullptr, {});
    REQUIRE(output->num_columns() == 3);
    auto const count = output->num_rows();
    std::vector<int64_t> rows(count), ids(count);
    std::vector<uint64_t> sources(count);
    if (count) {
      REQUIRE(cudaMemcpy(rows.data(),
                         output->view().column(0).data<int64_t>(),
                         count * sizeof(int64_t),
                         cudaMemcpyDeviceToHost) == cudaSuccess);
      REQUIRE(cudaMemcpy(sources.data(),
                         output->view().column(1).data<uint64_t>(),
                         count * sizeof(uint64_t),
                         cudaMemcpyDeviceToHost) == cudaSuccess);
      REQUIRE(cudaMemcpy(ids.data(),
                         output->view().column(2).data<int64_t>(),
                         count * sizeof(int64_t),
                         cudaMemcpyDeviceToHost) == cudaSuccess);
    }
    for (std::size_t i = 0; i < rows.size(); ++i) {
      REQUIRE((sources[i] == 0 || sources[i] == 2));
      REQUIRE(rows[i] >= 1000);
      REQUIRE(rows[i] < 6143);
      CHECK(rows[i] == ids[i]);
      CHECK((rows[i] < 2048 || rows[i] >= 4096));
      auto const key = sources[i] * 6144 + rows[i];
      CHECK_FALSE(seen[key]);
      seen[key] = true;
    }
    output_rows += count;
  }
  auto const surviving_sources = selection == 2 ? 0 : selection == 3 ? 1 : 2;
  // The reader drops the final row inside the selected groups. The virtual row-number
  // predicate remains residual and removes further rows after provenance assembly.
  CHECK(decoded_rows == surviving_sources * 4095);
  CHECK(output_rows == surviving_sources * 3095);
}

TEST_CASE_METHOD(carrier_file_fixture,
                 "parquet carriers normalize INTEGER and BIGINT files to INT8",
                 "[scan][parquet][carrier][provenance_mechanism]")
{
  std::vector<std::string> paths;
  {
    sirius::test::scoped_sirius_disable disable;
    duckdb::DuckDB db(nullptr);
    duckdb::Connection con(db);
    for (auto const* type : {"INTEGER", "BIGINT"}) {
      paths.push_back(scratch.file(std::string(type) + ".parquet"));
      auto result =
        con.Query("COPY (SELECT range::" + std::string(type) + " AS x FROM range(7)) TO " +
                  sirius::test::sql_literal(paths.back()) + " (FORMAT PARQUET)");
      REQUIRE(result);
      REQUIRE_FALSE(result->HasError());
    }
  }
  auto info                    = std::make_unique<scan::parquet_ingestible_table_info>();
  info->resolved_file_paths    = paths;
  info->names                  = {"x"};
  info->returned_types         = {sirius::logical_type::make(sirius::type_id::BIGINT)};
  info->sirius_dynamic_filters = std::make_shared<sirius::op::sirius_dynamic_filter_set>();
  auto const id                = duckdb::MultiFileReader::COLUMN_IDENTIFIER_FILE_ROW_NUMBER;
  info->column_ids             = {duckdb::ColumnIndex(id)};
  info->virtual_columns        = {
    {id, "file_row_number", sirius::logical_type::make(sirius::type_id::BIGINT), std::nullopt}};
  info->projected_output_count = 1;
  auto reader                  = scan::make_ingestible(std::move(info));
  std::vector<std::unique_ptr<scan::parquet_file_scan_info>> files;
  while (!reader->has_processed_all_metadata()) {
    files.push_back(read_file(*reader, 1));
  }
  auto splits = coalesce_files(*reader, std::move(files));
  REQUIRE(splits.size() == 2);
  auto manager      = sirius::test::operator_utils::initialize_memory_manager();
  auto const* space = manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
  REQUIRE(space);
  for (auto const& split : splits) {
    auto decoded = reader->materialize_metadata_to_table(
      *split, *space, cudf::get_default_stream(), false, nullptr);
    CHECK(decoded.table.view().num_rows() == 7);
    CHECK(decoded.table.view().column(0).type().id() == cudf::type_id::INT8);
    CHECK(decoded.table.view().column(1).type().id() == cudf::type_id::INT64);
  }
  CHECK(reader->reader_calls() == 2);
}

TEST_CASE("parquet carrier widths follow decoded temporal and integer annotations",
          "[scan][parquet][sizing][carrier]")
{
  using namespace cudf::io::parquet;
  SchemaElement column;
  column.parent_idx = 0;
  column.type       = Type::INT32;
  CHECK(scan::detail::carrier_decoded_width(column) == 4);
  column.converted_type = ConvertedType::TIME_MILLIS;
  CHECK(scan::detail::carrier_decoded_width(column) == 8);
  column.logical_type = LogicalType{TimeType{true, {TimeUnit::MILLIS}}};
  CHECK(scan::detail::carrier_decoded_width(column) == 8);
  column.logical_type = LogicalType{IntType{16, true}};
  CHECK(scan::detail::carrier_decoded_width(column) == 2);
  column.logical_type.reset();
  column.converted_type = ConvertedType::UINT_8;
  CHECK(scan::detail::carrier_decoded_width(column) == 1);
  column.converted_type = ConvertedType::INT_16;
  CHECK(scan::detail::carrier_decoded_width(column) == 2);
  column.converted_type.reset();
  column.type       = Type::INT64;
  column.arrow_type = cudf::type_id::DURATION_DAYS;
  CHECK(scan::detail::carrier_decoded_width(column) == 4);
  column.arrow_type = cudf::type_id::DURATION_MILLISECONDS;
  CHECK(scan::detail::carrier_decoded_width(column) == 8);
  column.logical_type = LogicalType{IntType{8, false}};
  CHECK(scan::detail::carrier_decoded_width(column) == 1);
  column.logical_type = LogicalType{DecimalType{0, 8}};
  CHECK(scan::detail::carrier_decoded_width(column) == 0);
  column.logical_type.reset();
  column.max_repetition_level = 1;
  CHECK(scan::detail::carrier_decoded_width(column) == 0);
}
