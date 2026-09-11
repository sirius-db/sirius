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

// test
#include <catch.hpp>

// sirius
#include <io/kvikio/kvikio_context.hpp>
#include <op/scan/parquet_byte_range.hpp>
#include <op/scan/parquet_gpu_ingestible.hpp>
#include <sirius/exception.hpp>

// cudf
#include <cudf/column/column_factories.hpp>
#include <cudf/io/datasource.hpp>
#include <cudf/io/experimental/hybrid_scan.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/parquet_io_utils.hpp>
#include <cudf/io/parquet_schema.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/default_stream.hpp>

// rmm
#include <rmm/device_buffer.hpp>

// standard library
#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <numeric>
#include <set>
#include <vector>

namespace fs = std::filesystem;

using sirius::op::scan::detail::row_group_start_offset;
using sirius::op::scan::detail::row_groups_in_byte_range;

namespace {

/// Metadata with one row group per given start offset, encoded the common way
/// (first column's data_page_offset carries the start; everything else unset).
cudf::io::parquet::FileMetaData make_metadata(std::vector<std::int64_t> const& rg_starts)
{
  cudf::io::parquet::FileMetaData meta;
  for (auto const start : rg_starts) {
    cudf::io::parquet::RowGroup rg;
    cudf::io::parquet::ColumnChunk chunk;
    chunk.meta_data.data_page_offset = start;
    rg.columns.push_back(std::move(chunk));
    meta.row_groups.push_back(std::move(rg));
  }
  return meta;
}

std::vector<cudf::size_type> all_indices(std::size_t n)
{
  std::vector<cudf::size_type> all(n);
  std::iota(all.begin(), all.end(), 0);
  return all;
}

fs::path lineitem_parquet_path()
{
#ifdef SIRIUS_PROJECT_ROOT
  return fs::path(SIRIUS_PROJECT_ROOT) / "test/cpp/integration/data/parquet/lineitem.parquet";
#else
  return fs::path(__FILE__).parent_path().parent_path() /
         "integration/data/parquet/lineitem.parquet";
#endif
}

}  // namespace

TEST_CASE("row_group_start_offset follows the StarRocks reader convention", "[parquet_byte_range]")
{
  cudf::io::parquet::FileMetaData meta;
  cudf::io::parquet::RowGroup rg;
  cudf::io::parquet::ColumnChunk chunk;

  SECTION("data page offset alone")
  {
    chunk.meta_data.data_page_offset = 100;
    rg.columns.push_back(chunk);
    meta.row_groups.push_back(rg);
    REQUIRE(row_group_start_offset(meta, 0) == 100);
  }

  SECTION("dictionary page precedes the data page")
  {
    chunk.meta_data.data_page_offset       = 100;
    chunk.meta_data.dictionary_page_offset = 40;
    rg.columns.push_back(chunk);
    meta.row_groups.push_back(rg);
    REQUIRE(row_group_start_offset(meta, 0) == 40);
  }

  SECTION("zero offsets count as absent, not as position zero")
  {
    chunk.meta_data.data_page_offset       = 100;
    chunk.meta_data.dictionary_page_offset = 0;
    chunk.meta_data.index_page_offset      = 0;
    rg.columns.push_back(chunk);
    meta.row_groups.push_back(rg);
    REQUIRE(row_group_start_offset(meta, 0) == 100);
  }

  SECTION("row group file_offset participates when set")
  {
    chunk.meta_data.data_page_offset = 100;
    rg.columns.push_back(chunk);
    rg.file_offset = 30;
    meta.row_groups.push_back(rg);
    REQUIRE(row_group_start_offset(meta, 0) == 30);
  }

  SECTION("no candidate offset at all is a loud error, never a guess")
  {
    rg.columns.push_back(chunk);  // all offsets zero
    meta.row_groups.push_back(rg);
    REQUIRE_THROWS_AS(row_group_start_offset(meta, 0), sirius::invalid_input_exception);
  }
}

TEST_CASE("any exact tiling reads every row group exactly once", "[parquet_byte_range]")
{
  // Row-group starts chosen so boundaries land before, on, and inside them.
  auto const starts    = std::vector<std::int64_t>{4, 1000, 1024, 5000, 999999};
  auto const meta      = make_metadata(starts);
  auto const file_size = std::uint64_t{1200000};

  // Sweep every tiling of the file into k equal ranges (the FE emits exact tilings).
  for (std::size_t k : {1, 2, 3, 4, 5, 7, 16}) {
    std::vector<cudf::size_type> combined;
    std::set<cudf::size_type> seen;
    auto const split = file_size / k;
    for (std::size_t part = 0; part < k; ++part) {
      auto const start  = part * split;
      auto const length = (part == k - 1) ? file_size - start : split;
      auto const owned  = row_groups_in_byte_range(meta, start, length);
      for (auto const idx : owned) {
        INFO("tiling k=" << k << " part=" << part);
        REQUIRE(seen.insert(idx).second);  // pairwise disjoint
        combined.push_back(idx);
      }
    }
    std::sort(combined.begin(), combined.end());
    INFO("tiling k=" << k);
    REQUIRE(combined == all_indices(starts.size()));  // complete
  }
}

TEST_CASE("byte-range ownership edge cases", "[parquet_byte_range]")
{
  auto const meta = make_metadata({4, 1000, 5000});

  SECTION("whole file owns everything")
  {
    REQUIRE(row_groups_in_byte_range(meta, 0, 6000) == all_indices(3));
  }

  SECTION("a straddling row group belongs to the range holding its start")
  {
    // Range ends at 1500, inside row group 1's bytes; rg 1 starts at 1000 -> owned here...
    REQUIRE(row_groups_in_byte_range(meta, 0, 1500) == std::vector<cudf::size_type>{0, 1});
    // ...and not by the neighbour that covers its tail.
    REQUIRE(row_groups_in_byte_range(meta, 1500, 3000) == std::vector<cudf::size_type>{});
  }

  SECTION("a range inside one row group owns nothing (valid empty split)")
  {
    REQUIRE(row_groups_in_byte_range(meta, 1200, 100).empty());
  }

  SECTION("zero length owns nothing, including the canonical empty split")
  {
    REQUIRE(row_groups_in_byte_range(meta, 0, 0).empty());
    REQUIRE(row_groups_in_byte_range(meta, 6000, 0).empty());
  }

  SECTION("a boundary exactly on a row-group start assigns it to the right-hand range")
  {
    REQUIRE(row_groups_in_byte_range(meta, 4, 996) == std::vector<cudf::size_type>{0});
    REQUIRE(row_groups_in_byte_range(meta, 1000, 5000) == std::vector<cudf::size_type>{1, 2});
  }
}

TEST_CASE("real footer: rule agrees with cudf's byte-range filter on the test lineitem",
          "[parquet_byte_range]")
{
  auto const path = lineitem_parquet_path();
  REQUIRE(fs::exists(path));
  auto source = cudf::io::datasource::create(path.string());
  auto footer = cudf::io::parquet::fetch_footer_to_host(*source);
  cudf::io::parquet_reader_options options;
  cudf::io::parquet::experimental::hybrid_scan_reader reader(
    cudf::host_span<uint8_t const>(footer->data(), footer->size()), options);
  auto const metadata = reader.parquet_metadata();
  REQUIRE(!metadata.row_groups.empty());
  auto const file_size = std::uint64_t(fs::file_size(path));

  // Exactly-once over a two-way tiling of the real file.
  auto const half = file_size / 2;
  auto left       = row_groups_in_byte_range(metadata, 0, half);
  auto right      = row_groups_in_byte_range(metadata, half, file_size - half);
  std::vector<cudf::size_type> combined = left;
  combined.insert(combined.end(), right.begin(), right.end());
  std::sort(combined.begin(), combined.end());
  REQUIRE(combined == all_indices(metadata.row_groups.size()));

  // Cross-check against cudf's own byte-range pruning. Informational: cudf's definition of a
  // row group's start is not pinned by its API docs; if this ever diverges, our rule (the
  // StarRocks reader convention) stays authoritative and this warning says cudf changed.
  auto all = reader.all_row_groups(options);
  cudf::io::parquet_reader_options range_options;
  range_options.set_skip_bytes(0);
  range_options.set_num_bytes(half);
  auto cudf_left = reader.filter_row_groups_with_byte_range(
    cudf::host_span<cudf::size_type const>(all.data(), all.size()), range_options);
  if (cudf_left != left) {
    WARN("cudf filter_row_groups_with_byte_range diverges from the StarRocks rule: cudf="
         << cudf_left.size() << " ours=" << left.size());
  }
}

namespace {

/// Writes a one-column (a BIGINT, values 0..rows-1) parquet with `rows / rows_per_group` row
/// groups, and returns its path. Deterministic row-group layout for the split-scan tests.
fs::path write_multi_row_group_parquet(fs::path const& dir,
                                       std::int64_t rows,
                                       std::int64_t rows_per_group)
{
  auto const path = dir / "byte_range_scan.parquet";
  auto stream     = cudf::get_default_stream();
  std::vector<std::int64_t> host(rows);
  std::iota(host.begin(), host.end(), 0);
  rmm::device_buffer data(host.data(), rows * sizeof(std::int64_t), stream);
  auto column = std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::INT64},
                                               static_cast<cudf::size_type>(rows),
                                               std::move(data),
                                               rmm::device_buffer{},
                                               0);
  cudf::table_view table({column->view()});
  cudf::io::table_input_metadata metadata(table);
  metadata.column_metadata[0].set_name("a");
  auto options =
    cudf::io::parquet_writer_options::builder(cudf::io::sink_info(path.string()), table)
      .metadata(std::move(metadata))
      .row_group_size_rows(rows_per_group)
      .build();
  cudf::io::write_parquet(options);
  return path;
}

/// Table info for a byte-range scan of `path` projecting the single BIGINT column.
std::unique_ptr<sirius::op::scan::parquet_ingestible_table_info> ranged_info(fs::path const& path,
                                                                             std::uint64_t start,
                                                                             std::uint64_t length)
{
  auto info                  = std::make_unique<sirius::op::scan::parquet_ingestible_table_info>();
  info->resolved_file_paths  = {path.string()};
  info->resolved_file_ranges = {{start, length}};
  info->names                = {"a"};
  info->returned_types.push_back(sirius::logical_type::make(sirius::type_id::BIGINT));
  info->column_ids.push_back(duckdb::ColumnIndex(0));
  info->scan_output_arity      = 1;
  info->approximate_batch_size = std::size_t{1} << 30;
  return info;
}

/// Drives the ingestible for one range and returns the selected row-group indices plus the
/// number of rows they hold (from the footer metadata — no decode needed).
std::pair<std::vector<cudf::size_type>, std::int64_t> scan_selection(
  std::unique_ptr<sirius::op::scan::parquet_ingestible_table_info> info)
{
  auto ingestible = sirius::op::scan::make_ingestible(std::move(info));
  auto ioctx      = std::make_shared<sirius::io::kvikio_context>();
  auto task       = ingestible->next_split_provider(
    [ioctx](std::string_view) -> std::shared_ptr<sirius::io::sirius_ioctx> { return ioctx; });
  REQUIRE(task);
  auto file = task();
  REQUIRE(file);

  // Splits materialize downstream of the coalescer, exactly as in production.
  auto coalescer = ingestible->create_batch_coalescer();
  auto batches   = coalescer->push(std::move(file));
  for (auto& batch : coalescer->flush()) {
    batches.push_back(std::move(batch));
  }

  std::vector<cudf::size_type> indices;
  std::int64_t rows = 0;
  for (auto const& batch : batches) {
    auto* split = dynamic_cast<sirius::op::scan::parquet_split_info*>(batch.get());
    REQUIRE(split);
    for (auto const& slice : split->rg_slices) {
      for (auto const idx : slice.row_group_indices) {
        indices.push_back(idx);
        rows += slice.file_metadata->row_groups.at(idx).num_rows;
      }
    }
  }
  std::sort(indices.begin(), indices.end());
  return {indices, rows};
}

}  // namespace

TEST_CASE("a two-way split scan selects disjoint, complete row groups",
          "[parquet_byte_range][scan]")
{
  auto const dir = fs::temp_directory_path() / "sirius_byte_range_test";
  fs::create_directories(dir);
  // cudf clamps row_group_size_rows to a 5000-row floor; 50k rows -> 10 real row groups of
  // sequential int64s, large enough that both halves of the file hold data pages.
  constexpr std::int64_t kRows = 50000;
  auto const path              = write_multi_row_group_parquet(dir, kRows, 5000);
  auto const file_size         = std::uint64_t(fs::file_size(path));
  auto const half              = file_size / 2;

  auto [left_rgs, left_rows]   = scan_selection(ranged_info(path, 0, half));
  auto [right_rgs, right_rows] = scan_selection(ranged_info(path, half, file_size - half));

  INFO("left row groups: " << left_rgs.size() << ", right: " << right_rgs.size());
  REQUIRE(!left_rgs.empty());
  REQUIRE(!right_rgs.empty());
  for (auto const idx : left_rgs) {
    REQUIRE(std::find(right_rgs.begin(), right_rgs.end(), idx) == right_rgs.end());
  }
  REQUIRE(left_rows + right_rows == kRows);

  // A whole-file scan ((0,0) range) is byte-identical to no range at all.
  auto [all_rgs, all_rows] = scan_selection(ranged_info(path, 0, 0));
  REQUIRE(all_rows == kRows);
  REQUIRE(std::int64_t(left_rgs.size() + right_rgs.size()) == std::int64_t(all_rgs.size()));

  // A range inside one row group is a valid empty scan.
  auto [none_rgs, none_rows] = scan_selection(ranged_info(path, 10, 5));
  REQUIRE(none_rgs.empty());
  REQUIRE(none_rows == 0);

  fs::remove_all(dir);
}

TEST_CASE("mismatched range/path pairing is refused at construction", "[parquet_byte_range][scan]")
{
  auto info                  = std::make_unique<sirius::op::scan::parquet_ingestible_table_info>();
  info->resolved_file_paths  = {"a.parquet", "b.parquet"};
  info->resolved_file_ranges = {{0, 10}};
  info->names                = {"a"};
  info->returned_types.push_back(sirius::logical_type::make(sirius::type_id::BIGINT));
  info->column_ids.push_back(duckdb::ColumnIndex(0));
  info->scan_output_arity = 1;
  REQUIRE_THROWS_AS(sirius::op::scan::make_ingestible(std::move(info)),
                    sirius::invalid_input_exception);
}
