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

// Unit tests for parquet_gpu_ingestible. These cover the pieces that were
// previously exercised by test_parquet_split_provider.cpp before step 10
// deleted the legacy provider:
//   - FLBA-decimal pushdown probe (DECIMAL(25,2) forces
//     FIXED_LEN_BYTE_ARRAY physical type; cudf's stats filter cannot
//     compare against a fixed_point_scalar AST literal, so the ingestible
//     must skip pushdown).
//   - INT64-decimal allow-pushdown (DECIMAL(10,2) fits in INT64; pushdown
//     should remain enabled).
//   - No-filter smoke (has_more_splits / next_split_provider drive a
//     batch end-to-end and emit one scan_operator_input per row-group
//     batch).

#include "catch.hpp"
#include "test_helpers_ioctx.hpp"

#include <duckdb.hpp>
#include <duckdb/common/types.hpp>
#include <duckdb/planner/expression.hpp>
#include <duckdb/planner/filter/constant_filter.hpp>
#include <duckdb/planner/table_filter.hpp>
#include <helper/logical_type.hpp>
#include <op/scan/parquet_gpu_ingestible.hpp>
#include <op/scan/sirius_gpu_scan_operator_data.hpp>
#include <scan_manager/sirius_scan_manager.hpp>

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace {

namespace sscan = sirius::op::scan;

// Drain an ingestible synchronously (no scheduler), collecting every
// emitted operator_data. The base split_provider::run loop is trivially
// reproducible by hand here — has_more_splits then claim then invoke —
// and avoids dragging in static_thread_pool just to test the claim/work
// behavior in isolation.
std::vector<std::unique_ptr<sirius::op::operator_data>> drain_ingestible(
  sscan::parquet_gpu_ingestible& ingestible)
{
  std::vector<std::unique_ptr<sirius::op::operator_data>> splits;
  while (ingestible.has_more_splits()) {
    auto work = ingestible.next_split_provider();
    if (!work) { continue; }
    auto batch = work();
    for (auto& s : batch) {
      splits.push_back(std::move(s));
    }
  }
  return splits;
}

// Write a parquet file with one INTEGER column and one DECIMAL(p,s) column.
// Returns the path. Caller owns the cleanup.
std::filesystem::path write_decimal_parquet(std::filesystem::path const& dir,
                                            std::string const& filename,
                                            int precision,
                                            int scale,
                                            std::size_t row_count,
                                            std::size_t row_group_size)
{
  std::error_code ec;
  std::filesystem::remove_all(dir, ec);
  std::filesystem::create_directories(dir);

  duckdb::DuckDB db(nullptr);
  duckdb::Connection con(db);

  std::string const table = "tmp_" + filename;
  auto result             = con.Query("CREATE TABLE " + table +
                          " AS SELECT (range)::INTEGER AS id, "
                                      "CAST(range * 1.25 AS DECIMAL(" +
                          std::to_string(precision) + "," + std::to_string(scale) +
                          ")) AS amount FROM range(0, " + std::to_string(row_count) + ")");
  REQUIRE(result);
  REQUIRE_FALSE(result->HasError());

  auto const path = dir / (filename + ".parquet");
  result          = con.Query("COPY " + table + " TO '" + path.string() +
                     "' (FORMAT PARQUET, COMPRESSION zstd, ROW_GROUP_SIZE " +
                     std::to_string(row_group_size) + ")");
  REQUIRE(result);
  REQUIRE_FALSE(result->HasError());
  con.Query("DROP TABLE " + table);

  return path;
}

// Build a parquet_ingestible_table_info for the two-column decimal fixture.
std::unique_ptr<sscan::parquet_ingestible_table_info> make_table_info(
  std::filesystem::path const& path,
  int precision,
  int scale,
  duckdb::unique_ptr<duckdb::TableFilterSet> filters)
{
  auto info            = std::make_unique<sscan::parquet_ingestible_table_info>();
  info->returned_types = {
    sirius::logical_type::make(sirius::type_id::INTEGER),
    sirius::logical_type::make_decimal(precision, scale),
  };
  info->resolved_file_paths    = {path.string()};
  info->column_ids             = {duckdb::ColumnIndex(0), duckdb::ColumnIndex(1)};
  info->projection_ids         = {};
  info->names                  = {"id", "amount"};
  info->table_filters          = std::move(filters);
  info->partition_indices      = {};
  info->scan_output_arity      = info->returned_types.size();
  info->approximate_batch_size = std::size_t{1} << 30;
  info->max_file_processed     = 10;
  return info;
}

duckdb::unique_ptr<duckdb::TableFilterSet> make_amount_lt_filter(int precision, int scale)
{
  // amount < 6250.00 — applied to the second column (index 1).
  auto filters = duckdb::make_uniq<duckdb::TableFilterSet>();
  filters->PushFilter(duckdb::ColumnIndex(1),
                      duckdb::make_uniq<duckdb::ConstantFilter>(
                        duckdb::ExpressionType::COMPARE_LESSTHAN,
                        duckdb::Value::DECIMAL(static_cast<int64_t>(625000), precision, scale)));
  return filters;
}

}  // namespace

TEST_CASE("parquet_gpu_ingestible - FLBA decimal disables filter pushdown",
          "[scan][parquet_gpu_ingestible][filter][flba_decimal]")
{
  // DECIMAL(25,2) forces DuckDB's parquet writer to use FIXED_LEN_BYTE_ARRAY
  // physical type (p > 18 exceeds INT64 capacity). cudf's row-group stats
  // filter throws "Invalid type and stats combination" comparing a
  // fixed_point_scalar against FLBA stats. The ingestible's schema probe
  // must detect this and set disable_filter_pushdown on every emitted
  // parquet_split_info. The filter still applies post-decode.
  auto const dir  = std::filesystem::temp_directory_path() / "pgi_flba_test";
  auto const path = write_decimal_parquet(dir,
                                          "flba",
                                          /*precision=*/25,
                                          /*scale=*/2,
                                          /*row_count=*/10000,
                                          /*row_group_size=*/5000);

  auto table_info = make_table_info(
    path, /*precision=*/25, /*scale=*/2, make_amount_lt_filter(/*precision=*/25, /*scale=*/2));

  sirius::scan_manager::sirius_scan_manager mgr({});
  auto gpu_ioctxs = sirius::scan_test_utils::make_test_gpu_ioctxs();
  sscan::parquet_gpu_ingestible ingestible(std::move(table_info), mgr, std::move(gpu_ioctxs));

  auto splits = drain_ingestible(ingestible);
  REQUIRE_FALSE(splits.empty());
  for (auto const& split : splits) {
    auto* input = dynamic_cast<sscan::scan_operator_input*>(split.get());
    REQUIRE(input != nullptr);
    auto const* parquet_info =
      dynamic_cast<sscan::parquet_split_info const*>(&input->metadata->scan());
    REQUIRE(parquet_info != nullptr);
    INFO("Every split from an FLBA-decimal file must have disable_filter_pushdown set");
    REQUIRE(parquet_info->disable_filter_pushdown);
  }

  std::filesystem::remove_all(dir);
}

TEST_CASE("parquet_gpu_ingestible - INT64 decimal allows filter pushdown",
          "[scan][parquet_gpu_ingestible][filter][flba_decimal]")
{
  // DECIMAL(10,2) fits in INT64 physical type — the probe must NOT disable
  // filter pushdown. Complement of the FLBA test above.
  auto const dir  = std::filesystem::temp_directory_path() / "pgi_int64dec_test";
  auto const path = write_decimal_parquet(dir,
                                          "int64dec",
                                          /*precision=*/10,
                                          /*scale=*/2,
                                          /*row_count=*/10000,
                                          /*row_group_size=*/5000);

  auto table_info = make_table_info(
    path, /*precision=*/10, /*scale=*/2, make_amount_lt_filter(/*precision=*/10, /*scale=*/2));

  sirius::scan_manager::sirius_scan_manager mgr({});
  auto gpu_ioctxs = sirius::scan_test_utils::make_test_gpu_ioctxs();
  sscan::parquet_gpu_ingestible ingestible(std::move(table_info), mgr, std::move(gpu_ioctxs));

  auto splits = drain_ingestible(ingestible);
  REQUIRE_FALSE(splits.empty());
  for (auto const& split : splits) {
    auto* input = dynamic_cast<sscan::scan_operator_input*>(split.get());
    REQUIRE(input != nullptr);
    auto const* parquet_info =
      dynamic_cast<sscan::parquet_split_info const*>(&input->metadata->scan());
    REQUIRE(parquet_info != nullptr);
    INFO("INT64-decimal files should allow filter pushdown");
    REQUIRE_FALSE(parquet_info->disable_filter_pushdown);
  }

  std::filesystem::remove_all(dir);
}

TEST_CASE("parquet_gpu_ingestible - no-filter scan emits a single scan_operator_input",
          "[scan][parquet_gpu_ingestible]")
{
  // Sanity-check the bare construct → drive → drain path with no filters
  // attached. Confirms that has_more_splits / next_split_provider produce
  // a scan_operator_input whose metadata carries a parquet_split_info
  // with no disable_filter_pushdown flag (no filter expression means the
  // pushdown probe never runs).
  auto const dir  = std::filesystem::temp_directory_path() / "pgi_nofilter_test";
  auto const path = write_decimal_parquet(dir,
                                          "nofilter",
                                          /*precision=*/10,
                                          /*scale=*/2,
                                          /*row_count=*/2000,
                                          /*row_group_size=*/1000);

  auto table_info = make_table_info(path,
                                    /*precision=*/10,
                                    /*scale=*/2,
                                    /*filters=*/nullptr);

  sirius::scan_manager::sirius_scan_manager mgr({});
  auto gpu_ioctxs = sirius::scan_test_utils::make_test_gpu_ioctxs();
  sscan::parquet_gpu_ingestible ingestible(std::move(table_info), mgr, std::move(gpu_ioctxs));

  auto splits = drain_ingestible(ingestible);
  REQUIRE_FALSE(splits.empty());

  auto* input = dynamic_cast<sscan::scan_operator_input*>(splits.front().get());
  REQUIRE(input != nullptr);
  REQUIRE(input->metadata != nullptr);
  REQUIRE_FALSE(input->metadata->has_filter());  // no filter → no post_filter_and_project info

  auto const* parquet_info =
    dynamic_cast<sscan::parquet_split_info const*>(&input->metadata->scan());
  REQUIRE(parquet_info != nullptr);
  REQUIRE_FALSE(parquet_info->disable_filter_pushdown);
  REQUIRE(parquet_info->reader_options != nullptr);
  REQUIRE(parquet_info->plan != nullptr);
  REQUIRE_FALSE(parquet_info->rg_slices.empty());

  std::filesystem::remove_all(dir);
}
