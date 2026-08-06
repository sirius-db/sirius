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
#include <op/scan/parquet_byte_range.hpp>
#include <sirius/exception.hpp>

// cudf
#include <cudf/io/datasource.hpp>
#include <cudf/io/experimental/hybrid_scan.hpp>
#include <cudf/io/parquet_io_utils.hpp>
#include <cudf/io/parquet_schema.hpp>

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
