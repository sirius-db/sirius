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
 * @file test_parquet_decimal_pushdown.cpp
 * @brief Reader-side filter pushdown over the decimal physical encodings parquet uses
 *
 * A decimal column is stored in parquet as INT32, INT64 or FIXED_LEN_BYTE_ARRAY depending on its
 * precision, and `parquet_gpu_ingestible` decides per file whether the filter may be handed to the
 * cuDF reader for row-group pruning. That decision used to disable pushdown for a whole file
 * whenever any scanned decimal was FIXED_LEN_BYTE_ARRAY, because cuDF's row-group statistics
 * filter threw comparing a `fixed_point_scalar` literal against those statistics. cuDF now decodes
 * them, so pushdown stays enabled -- which matters because DuckDB stores any DECIMAL wider than 18
 * digits that way and Arrow-based writers store every decimal that way.
 *
 * These tests pin both halves of that behaviour: that pushdown is not disabled, and that the
 * pruning it enables is actually correct. The second half is the one that matters if cuDF
 * regresses -- a decoding bug there would silently return wrong rows rather than throw, so the
 * fixture is built so that correct pruning is only possible when negative values sign-extend
 * correctly, which is the mechanism at issue.
 *
 * Sign extension happens per stored width, so the FIXED_LEN_BYTE_ARRAY cases cover 4, 8 and
 * 16 bytes. Only the 16-byte width can be produced by a file these tests write for themselves,
 * because DuckDB stores every decimal wider than 18 digits in 16 bytes whatever its precision;
 * the narrower widths come from checked-in pyarrow-written fixtures under `data/`.
 *
 * Two things here are NOT covered. The BYTE_ARRAY decimal encoding has no fixture because no
 * writer available here emits it, so the arm of the probe that still disables pushdown for it is
 * unverified rather than known-good. And these tests assert on row-group metadata only -- that the
 * right row groups survive -- which cannot catch a defect that prunes correctly but decodes the
 * surviving rows wrongly. `test/sql/parquet_decimal_pushdown.test` covers the decoded values.
 */

#include <catch.hpp>
#include <duckdb.hpp>
#include <duckdb/common/constants.hpp>
#include <duckdb/planner/filter/constant_filter.hpp>
#include <duckdb/planner/table_filter.hpp>
#include <io/kvikio/kvikio_context.hpp>
#include <op/scan/parquet_gpu_ingestible.hpp>
#include <op/scan/scan_plan.hpp>
#include <op/scan/sirius_gpu_scan_operator_data.hpp>
#include <unistd.h>

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace {

namespace scan = sirius::op::scan;

/// Rows per row group in the fixture, and the number of row groups it produces.
constexpr int64_t ROWS_PER_GROUP = 2048;
constexpr int64_t GROUP_COUNT    = 10;
/// Row groups 0..4 hold only negative amounts; 5..9 hold only non-negative ones.
constexpr std::size_t NEGATIVE_GROUP_COUNT = 5;

/// One decimal width to exercise, with the parquet physical type DuckDB stores it as.
struct decimal_encoding {
  int precision;
  int scale;
  char const* physical_type;
};

/// Precisions up to 9 fit INT32 and up to 18 fit INT64. DuckDB stores every wider precision as a
/// 16-byte FIXED_LEN_BYTE_ARRAY -- DECIMAL(25,2) and DECIMAL(38,4) differ in precision and scale
/// but not in stored width, so narrower FIXED_LEN_BYTE_ARRAY widths need the checked-in fixtures
/// below rather than a file this test can write for itself.
constexpr decimal_encoding ENCODINGS[] = {
  {9, 2, "INT32"},
  {18, 2, "INT64"},
  {25, 2, "FIXED_LEN_BYTE_ARRAY(16)"},
  {38, 4, "FIXED_LEN_BYTE_ARRAY(16)"},
};

/// A checked-in fixture pinning one FIXED_LEN_BYTE_ARRAY width, written by pyarrow, which sizes
/// the stored field to the precision instead of always using 16 bytes.
/// See `test/cpp/scan/data/generate_flba_decimal_bands.py`.
struct flba_fixture {
  char const* file_name;
  int precision;
  int scale;
  int byte_width;
};

constexpr flba_fixture FLBA_FIXTURES[] = {
  {"flba4_decimal_bands.parquet", 9, 2, 4},
  {"flba8_decimal_bands.parquet", 18, 2, 8},
  {"flba16_decimal_bands.parquet", 38, 4, 16},
};

std::filesystem::path project_root()
{
#ifdef SIRIUS_PROJECT_ROOT
  return std::filesystem::path{SIRIUS_PROJECT_ROOT};
#else
  return std::filesystem::current_path();
#endif
}

std::filesystem::path fixture_path(char const* file_name)
{
  return project_root() / "test/cpp/scan/data" / file_name;
}

/// Owns one test's fixture directory: cleared and created on construction, removed on
/// destruction, so a failing CHECK/REQUIRE cannot leak the directory past the test.
class scoped_fixture_dir {
 public:
  explicit scoped_fixture_dir(std::filesystem::path dir) : _dir(std::move(dir))
  {
    std::error_code ec;
    std::filesystem::remove_all(_dir, ec);
    std::filesystem::create_directories(_dir);
  }
  ~scoped_fixture_dir()
  {
    std::error_code ec;
    std::filesystem::remove_all(_dir, ec);
  }

  scoped_fixture_dir(scoped_fixture_dir const&)            = delete;
  scoped_fixture_dir& operator=(scoped_fixture_dir const&) = delete;

  [[nodiscard]] std::filesystem::path const& path() const noexcept { return _dir; }

 private:
  std::filesystem::path _dir;
};

/**
 * @brief Write a decimal parquet file whose row groups hold disjoint, mostly negative bands into
 * an existing directory (a `scoped_fixture_dir` owns it)
 *
 * Row group `g` holds amounts in `[(g - 5) * 10000, (g - 5) * 10000 + 2047]`, so the bands never
 * overlap and the first five are entirely negative. Pruning a predicate that selects only negative
 * amounts must therefore keep exactly those five, which it can only do if the statistics for a
 * negative band are decoded with their sign intact -- read as unsigned, a negative minimum becomes
 * a huge positive and the group is pruned away, losing rows.
 */
std::filesystem::path write_banded_decimal_parquet(std::filesystem::path const& dir,
                                                   std::string const& name,
                                                   int precision,
                                                   int scale)
{
  duckdb::DuckDB db(nullptr);
  duckdb::Connection con(db);

  auto const decimal_type =
    "DECIMAL(" + std::to_string(precision) + "," + std::to_string(scale) + ")";
  auto const row_count = ROWS_PER_GROUP * GROUP_COUNT;
  auto const table     = "banded_" + name;

  // Every 97th row is NULL so the fixture also covers null handling in the statistics.
  auto result = con.Query(
    "CREATE TABLE " + table +
    " AS SELECT (range)::INTEGER AS id, CASE WHEN range % 97 = 0 THEN NULL ELSE CAST(((range // " +
    std::to_string(ROWS_PER_GROUP) + ") - " + std::to_string(NEGATIVE_GROUP_COUNT) +
    ") * 10000 + (range % " + std::to_string(ROWS_PER_GROUP) + ") AS " + decimal_type +
    ") END AS amount FROM range(0, " + std::to_string(row_count) + ")");
  REQUIRE(result);
  REQUIRE_FALSE(result->HasError());

  auto const path = dir / (name + ".parquet");
  result          = con.Query("COPY " + table + " TO '" + path.string() +
                     "' (FORMAT PARQUET, COMPRESSION zstd, ROW_GROUP_SIZE " +
                     std::to_string(ROWS_PER_GROUP) + ")");
  REQUIRE(result);
  REQUIRE_FALSE(result->HasError());

  return path;
}

/// Build `amount < -5` as a DuckDB table filter on the second column.
duckdb::unique_ptr<duckdb::TableFilterSet> make_negative_amount_filter(int precision, int scale)
{
  int64_t scale_factor = 1;
  for (int i = 0; i < scale; ++i) {
    scale_factor *= 10;
  }
  auto const raw = static_cast<int64_t>(-5) * scale_factor;
  auto filters   = duckdb::make_uniq<duckdb::TableFilterSet>();
  filters->PushFilter(
    duckdb::ColumnIndex(1),
    duckdb::make_uniq<duckdb::ConstantFilter>(
      duckdb::ExpressionType::COMPARE_LESSTHAN,
      duckdb::Value::DECIMAL(raw, static_cast<uint8_t>(precision), static_cast<uint8_t>(scale))));
  return filters;
}

std::unique_ptr<scan::parquet_ingestible_table_info> make_table_info(
  std::filesystem::path const& path,
  int precision,
  int scale,
  duckdb::unique_ptr<duckdb::TableFilterSet> filters)
{
  auto info            = std::make_unique<scan::parquet_ingestible_table_info>();
  info->returned_types = {
    sirius::logical_type::make(sirius::type_id::INTEGER),
    sirius::logical_type::make_decimal(static_cast<uint8_t>(precision),
                                       static_cast<uint8_t>(scale)),
  };
  info->resolved_file_paths    = {path.string()};
  info->column_ids             = {duckdb::ColumnIndex(0), duckdb::ColumnIndex(1)};
  info->names                  = {"id", "amount"};
  info->table_filters          = std::move(filters);
  info->scan_output_arity      = info->returned_types.size();
  info->approximate_batch_size = std::size_t{1} << 30;
  return info;
}

/// What one metadata scan of the fixture reveals about the pushdown decision and its result.
struct scan_outcome {
  bool disable_filter_pushdown;
  std::vector<cudf::size_type> surviving_row_groups;
  int64_t surviving_rows;
};

scan_outcome scan_file(std::filesystem::path const& path,
                       int precision,
                       int scale,
                       bool with_filter)
{
  auto info = make_table_info(
    path, precision, scale, with_filter ? make_negative_amount_filter(precision, scale) : nullptr);

  auto ingestible = scan::make_ingestible(std::move(info));
  auto ioctx      = std::make_shared<sirius::io::kvikio_context>();
  auto task       = ingestible->next_split_provider(
    [ioctx](std::string_view) -> std::shared_ptr<sirius::io::sirius_ioctx> { return ioctx; });
  REQUIRE(task);

  auto file = task();
  REQUIRE(file);
  auto* file_scan = dynamic_cast<scan::parquet_file_scan_info*>(file.get());
  REQUIRE(file_scan != nullptr);

  scan_outcome outcome{file_scan->disable_filter_pushdown, {}, 0};
  for (auto const& entry : file_scan->row_groups) {
    outcome.surviving_row_groups.push_back(entry.index);
    outcome.surviving_rows += entry.num_rows;
  }
  return outcome;
}

}  // namespace

TEST_CASE("parquet decimal pushdown - every DuckDB decimal encoding keeps pushdown enabled",
          "[scan][parquet][filter][decimal]")
{
  for (auto const& encoding : ENCODINGS) {
    CAPTURE(encoding.precision, encoding.scale, encoding.physical_type);
    scoped_fixture_dir const dir(std::filesystem::temp_directory_path() /
                                 ("pgi_decimal_pushdown_p" + std::to_string(encoding.precision) +
                                  "." + std::to_string(::getpid())));
    auto const path =
      write_banded_decimal_parquet(dir.path(), "amounts", encoding.precision, encoding.scale);

    auto const outcome = scan_file(path, encoding.precision, encoding.scale, /*with_filter=*/true);

    INFO("A decimal column must not disable reader-side filter pushdown for its whole file");
    CHECK_FALSE(outcome.disable_filter_pushdown);
  }
}

TEST_CASE("parquet decimal pushdown - row groups holding only negative amounts survive pruning",
          "[scan][parquet][filter][decimal]")
{
  for (auto const& encoding : ENCODINGS) {
    CAPTURE(encoding.precision, encoding.scale, encoding.physical_type);
    scoped_fixture_dir const dir(std::filesystem::temp_directory_path() /
                                 ("pgi_decimal_prune_p" + std::to_string(encoding.precision) + "." +
                                  std::to_string(::getpid())));
    auto const path =
      write_banded_decimal_parquet(dir.path(), "amounts", encoding.precision, encoding.scale);

    auto const unfiltered = scan_file(path, encoding.precision, encoding.scale, false);
    REQUIRE(unfiltered.surviving_row_groups.size() == static_cast<std::size_t>(GROUP_COUNT));

    auto const filtered = scan_file(path, encoding.precision, encoding.scale, true);

    // `amount < -5` can only match the five all-negative bands. Keeping more means the statistics
    // did not prune; keeping fewer means rows that satisfy the predicate were dropped.
    INFO("Pruning must keep exactly the row groups whose amounts are negative");
    CHECK(filtered.surviving_row_groups == std::vector<cudf::size_type>{0, 1, 2, 3, 4});
    CHECK(filtered.surviving_rows == static_cast<int64_t>(NEGATIVE_GROUP_COUNT) * ROWS_PER_GROUP);
  }
}

TEST_CASE("parquet decimal pushdown - every FIXED_LEN_BYTE_ARRAY width prunes on negative amounts",
          "[scan][parquet][filter][decimal]")
{
  // Sign extension is applied per stored width, so a defect can exist at one width and not
  // another. The tests above can only reach the 16-byte width, because that is the only one
  // DuckDB emits; these fixtures carry the 4- and 8-byte widths as well.
  for (auto const& fixture : FLBA_FIXTURES) {
    CAPTURE(fixture.file_name, fixture.precision, fixture.scale, fixture.byte_width);
    auto const path = fixture_path(fixture.file_name);
    REQUIRE(std::filesystem::exists(path));

    auto const unfiltered = scan_file(path, fixture.precision, fixture.scale, false);
    REQUIRE(unfiltered.surviving_row_groups.size() == static_cast<std::size_t>(GROUP_COUNT));

    auto const filtered = scan_file(path, fixture.precision, fixture.scale, true);

    INFO("A FIXED_LEN_BYTE_ARRAY decimal must not disable pushdown at any stored width");
    CHECK_FALSE(filtered.disable_filter_pushdown);
    INFO("Pruning must keep exactly the row groups whose amounts are negative");
    CHECK(filtered.surviving_row_groups == std::vector<cudf::size_type>{0, 1, 2, 3, 4});
    CHECK(filtered.surviving_rows == static_cast<int64_t>(NEGATIVE_GROUP_COUNT) * ROWS_PER_GROUP);
  }
}
