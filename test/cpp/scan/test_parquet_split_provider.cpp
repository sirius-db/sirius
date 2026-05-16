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

// test
#include <catch.hpp>
#include <utils/utils.hpp>

// sirius
#include <exec/thread_pool.hpp>
#include <helper/logical_type.hpp>
#include <op/scan/parquet_scan_operator_data.hpp>
#include <scan_manager/parquet_split_provider.hpp>
#include <scan_manager/split_connector.hpp>

// duckdb
#include <duckdb.hpp>
#include <duckdb/common/column_index.hpp>
#include <duckdb/common/types.hpp>
#include <duckdb/common/vector.hpp>

// standard library
#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

using namespace sirius;
using namespace sirius::op::scan;
using namespace sirius::scan_manager;

namespace {

// Write a single parquet file with N rows and a fixed row-group size. Each row carries a few
// scalar columns so a row group has non-trivial uncompressed bytes.
std::filesystem::path write_parquet_file(duckdb::Connection& con,
                                         std::filesystem::path const& dir,
                                         std::string const& name,
                                         std::size_t num_rows,
                                         std::size_t row_group_size,
                                         std::int64_t start_id)
{
  std::filesystem::create_directories(dir);
  std::string const table = "psp_tmp_" + name;
  std::string const create_sql =
    "CREATE OR REPLACE TABLE " + table +
    " AS SELECT (range)::INTEGER AS id, ((range)*100)::BIGINT AS value, "
    "((range)*1.5)::DOUBLE AS price, ('item_' || range) AS name "
    "FROM range(" +
    std::to_string(start_id) + ", " +
    std::to_string(start_id + static_cast<std::int64_t>(num_rows)) + ")";
  auto result = con.Query(create_sql);
  REQUIRE(result);
  REQUIRE(!result->HasError());

  auto const path            = dir / (name + ".parquet");
  std::string const copy_sql = "COPY " + table + " TO '" + path.string() +
                               "' (FORMAT PARQUET, COMPRESSION zstd, ROW_GROUP_SIZE " +
                               std::to_string(row_group_size) + ")";
  result = con.Query(copy_sql);
  REQUIRE(result);
  REQUIRE(!result->HasError());

  result = con.Query("DROP TABLE " + table);
  REQUIRE(result);
  REQUIRE(!result->HasError());
  return path;
}

duckdb::vector<duckdb::ColumnIndex> all_column_ids(std::size_t n)
{
  duckdb::vector<duckdb::ColumnIndex> ids;
  ids.reserve(n);
  for (std::size_t i = 0; i < n; ++i) {
    ids.emplace_back(duckdb::ColumnIndex(i));
  }
  return ids;
}

duckdb::vector<sirius::logical_type> default_returned_types()
{
  return {
    sirius::logical_type::make(sirius::type_id::INTEGER),
    sirius::logical_type::make(sirius::type_id::BIGINT),
    sirius::logical_type::make(sirius::type_id::DOUBLE),
    sirius::logical_type::make(sirius::type_id::VARCHAR),
  };
}

// Drain every parquet_scan_data the provider emits.
std::vector<std::unique_ptr<parquet_scan_data>> drive_provider(parquet_split_provider& provider,
                                                               int n_threads = 2)
{
  exec::static_thread_pool pool(n_threads, "psp_test_pool");
  split_connector connector;
  // run() is fire-and-forget; metadata-scan errors flow through connector.close()
  // and surface here when get_next_split() rethrows.
  provider.run(pool, connector);

  std::vector<std::unique_ptr<parquet_scan_data>> drained;
  while (true) {
    auto next = connector.get_next_split();
    if (!next.has_value()) { break; }
    std::unique_ptr<op::operator_data> base = std::move(*next);
    auto* raw                               = base.release();
    auto* parquet                           = dynamic_cast<parquet_scan_data*>(raw);
    REQUIRE(parquet != nullptr);
    drained.emplace_back(std::unique_ptr<parquet_scan_data>(parquet));
  }
  return drained;
}

std::size_t distinct_files_in(parquet_scan_data const& data)
{
  std::unordered_set<std::string> files;
  for (auto const& slice : data.rg_slices) {
    files.insert(slice.file_path);
  }
  return files.size();
}

std::size_t total_uncompressed_bytes(parquet_scan_data const& data)
{
  std::size_t total = 0;
  for (auto const& slice : data.rg_slices) {
    total += slice.reserved_uncompressed_bytes;
  }
  return total;
}

// Helper to build a temp directory keyed by test name; cleared if it already exists.
std::filesystem::path fresh_tmp_dir(std::string const& tag)
{
  auto dir = std::filesystem::temp_directory_path() / ("psp_test_" + tag);
  std::error_code ec;
  std::filesystem::remove_all(dir, ec);
  std::filesystem::create_directories(dir);
  return dir;
}

}  // namespace

TEST_CASE("parquet_split_provider - max_file_processed bounds files per emitted split",
          "[scan_manager][parquet_split_provider]")
{
  // Generate 8 small files, set a tiny max_file_processed and a generous byte budget.
  // No single accumulated push should ever bundle more files than the cap allows.
  auto const dir       = fresh_tmp_dir("max_files");
  auto [db_owner, con] = sirius::make_test_db_and_connection();

  constexpr std::size_t k_files         = 8;
  constexpr std::size_t k_rows_per_file = 1000;
  constexpr std::size_t k_rg_size       = 1000;  // one row group per file

  std::vector<std::string> file_paths;
  file_paths.reserve(k_files);
  for (std::size_t i = 0; i < k_files; ++i) {
    auto path = write_parquet_file(con,
                                   dir,
                                   "f" + std::to_string(i),
                                   k_rows_per_file,
                                   k_rg_size,
                                   static_cast<std::int64_t>(i * k_rows_per_file));
    file_paths.push_back(path.string());
  }

  auto const returned_types = default_returned_types();

  constexpr std::size_t k_max_files_per_task = 2;
  constexpr std::size_t k_huge_batch_bytes   = std::size_t{1} << 30;  // effectively no byte cap

  parquet_split_provider provider(returned_types,
                                  file_paths,
                                  all_column_ids(returned_types.size()),
                                  /*projection_ids*/ {},
                                  /*names*/ {},
                                  /*scan_output_arity*/ returned_types.size(),
                                  /*table_filter_set*/ nullptr,
                                  /*partition_indices*/ {},
                                  k_huge_batch_bytes,
                                  k_max_files_per_task);

  auto splits = drive_provider(provider);
  REQUIRE_FALSE(splits.empty());

  // Every accumulated split must respect the file-count limit.
  for (auto const& split : splits) {
    auto const files = distinct_files_in(*split);
    INFO("split bundles " << files << " distinct files; cap is " << k_max_files_per_task);
    REQUIRE(files <= k_max_files_per_task);
  }

  // And every input file must appear somewhere across the emitted splits.
  std::unordered_set<std::string> seen;
  for (auto const& split : splits) {
    for (auto const& slice : split->rg_slices) {
      seen.insert(slice.file_path);
    }
  }
  REQUIRE(seen.size() == k_files);

  std::filesystem::remove_all(dir);
}

TEST_CASE("parquet_split_provider - max_file_processed=1 emits one file per split",
          "[scan_manager][parquet_split_provider]")
{
  // The strictest setting: each file should land in its own parquet_scan_data, regardless of
  // byte budget. Verifies that the per-task batching cap is propagated as a per-push cap.
  auto const dir       = fresh_tmp_dir("max_files_one");
  auto [db_owner, con] = sirius::make_test_db_and_connection();

  constexpr std::size_t k_files = 5;

  std::vector<std::string> file_paths;
  file_paths.reserve(k_files);
  for (std::size_t i = 0; i < k_files; ++i) {
    auto path = write_parquet_file(
      con, dir, "f" + std::to_string(i), 500, 500, static_cast<std::int64_t>(i * 500));
    file_paths.push_back(path.string());
  }

  auto const returned_types = default_returned_types();

  parquet_split_provider provider(returned_types,
                                  file_paths,
                                  all_column_ids(returned_types.size()),
                                  /*projection_ids*/ {},
                                  /*names*/ {},
                                  /*scan_output_arity*/ returned_types.size(),
                                  /*table_filter_set*/ nullptr,
                                  /*partition_indices*/ {},
                                  /*approximate_batch_size*/ std::size_t{1} << 30,
                                  /*max_file_processed*/ 1);

  auto splits = drive_provider(provider);
  REQUIRE(splits.size() == k_files);
  for (auto const& split : splits) {
    REQUIRE(distinct_files_in(*split) == 1);
  }

  std::filesystem::remove_all(dir);
}

TEST_CASE(
  "parquet_split_provider - approximate_batch_size flushes when accumulated bytes cross threshold",
  "[scan_manager][parquet_split_provider]")
{
  // Generate several files where each individual file's uncompressed size fits within the
  // budget, but two files' worth of bytes exceeds it. With the bug, accum.total_uncompressed_bytes
  // is never updated, so the provider bundles all files in a batch into one split, blowing past
  // the budget. After the fix, the accumulated bytes flush each time the budget is crossed.
  auto const dir       = fresh_tmp_dir("batch_size");
  auto [db_owner, con] = sirius::make_test_db_and_connection();

  constexpr std::size_t k_files         = 4;
  constexpr std::size_t k_rows_per_file = 5000;
  constexpr std::size_t k_rg_size       = 5000;  // single row group per file

  std::vector<std::string> file_paths;
  file_paths.reserve(k_files);
  for (std::size_t i = 0; i < k_files; ++i) {
    auto path = write_parquet_file(con,
                                   dir,
                                   "f" + std::to_string(i),
                                   k_rows_per_file,
                                   k_rg_size,
                                   static_cast<std::int64_t>(i * k_rows_per_file));
    file_paths.push_back(path.string());
  }

  auto const returned_types = default_returned_types();

  // Probe the file size by running with an effectively unlimited budget and a 1-file batch
  // cap so each split holds exactly one file's bytes.
  std::size_t per_file_bytes = 0;
  {
    parquet_split_provider probe(returned_types,
                                 file_paths,
                                 all_column_ids(returned_types.size()),
                                 /*projection_ids*/ {},
                                 /*names*/ {},
                                 /*scan_output_arity*/ returned_types.size(),
                                 /*table_filter_set*/ nullptr,
                                 /*partition_indices*/ {},
                                 /*approximate_batch_size*/ std::size_t{1} << 30,
                                 /*max_file_processed*/ 1);
    auto probe_splits = drive_provider(probe);
    REQUIRE(probe_splits.size() == k_files);
    for (auto const& split : probe_splits) {
      per_file_bytes = std::max(per_file_bytes, total_uncompressed_bytes(*split));
    }
  }
  REQUIRE(per_file_bytes > 0);

  // Set the byte budget below 2 files' worth so two-file bundles are forbidden, but allow
  // a generous max_file_processed so byte budget is the only gate. Drive everything through a
  // single task so accumulator state is exercised across files within one run_batch.
  std::size_t const budget_bytes = per_file_bytes + per_file_bytes / 2;  // ~1.5x one file

  parquet_split_provider provider(returned_types,
                                  file_paths,
                                  all_column_ids(returned_types.size()),
                                  /*projection_ids*/ {},
                                  /*names*/ {},
                                  /*scan_output_arity*/ returned_types.size(),
                                  /*table_filter_set*/ nullptr,
                                  /*partition_indices*/ {},
                                  budget_bytes,
                                  /*max_file_processed*/ k_files);  // one task covers all files

  auto splits = drive_provider(provider);
  REQUIRE_FALSE(splits.empty());

  // Each emitted split should hold roughly one file. Specifically, since the budget is below
  // 2 files of bytes, no split should contain 2 or more files.
  for (auto const& split : splits) {
    auto const files = distinct_files_in(*split);
    auto const bytes = total_uncompressed_bytes(*split);
    INFO("split bundles " << files << " files, " << bytes << " bytes (budget " << budget_bytes
                          << ", per-file " << per_file_bytes << ")");
    REQUIRE(files <= 1);
    // The seal-on-overflow rule allows a single trailing row group to push the total over the
    // budget. With a single-row-group-per-file layout the per-split bytes upper bound is
    // exactly one file's bytes.
    REQUIRE(bytes <= per_file_bytes);
  }

  // Sanity: every input file is still represented somewhere.
  std::unordered_set<std::string> seen;
  for (auto const& split : splits) {
    for (auto const& slice : split->rg_slices) {
      seen.insert(slice.file_path);
    }
  }
  REQUIRE(seen.size() == k_files);

  std::filesystem::remove_all(dir);
}

TEST_CASE(
  "parquet_split_provider - approximate_batch_size flushes mid-file across multiple row groups",
  "[scan_manager][parquet_split_provider]")
{
  // A single file with many row groups, and a budget that is below ~3 row groups' bytes.
  // Verifies the within-file overflow path: each emitted split's bytes stay near the budget
  // rather than carrying the whole file into one push.
  auto const dir       = fresh_tmp_dir("batch_size_intra_file");
  auto [db_owner, con] = sirius::make_test_db_and_connection();

  constexpr std::size_t k_rg_size          = 1000;
  constexpr std::size_t k_rows_in_one_file = k_rg_size * 8;  // 8 row groups

  auto path = write_parquet_file(con, dir, "single", k_rows_in_one_file, k_rg_size, 0);

  auto const returned_types = default_returned_types();

  // Probe the per-row-group bytes.
  std::size_t whole_file_bytes = 0;
  {
    parquet_split_provider probe(returned_types,
                                 {path.string()},
                                 all_column_ids(returned_types.size()),
                                 /*projection_ids*/ {},
                                 /*names*/ {},
                                 /*scan_output_arity*/ returned_types.size(),
                                 /*table_filter_set*/ nullptr,
                                 /*partition_indices*/ {},
                                 /*approximate_batch_size*/ std::size_t{1} << 30,
                                 /*max_file_processed*/ 1);
    auto probe_splits = drive_provider(probe);
    REQUIRE(probe_splits.size() == 1);
    whole_file_bytes = total_uncompressed_bytes(*probe_splits.front());
  }
  REQUIRE(whole_file_bytes > 0);

  // Pick a budget around 3 row groups' worth.
  std::size_t const per_rg_bytes = whole_file_bytes / 8;
  std::size_t const budget_bytes = per_rg_bytes * 3;

  parquet_split_provider provider(returned_types,
                                  {path.string()},
                                  all_column_ids(returned_types.size()),
                                  /*projection_ids*/ {},
                                  /*names*/ {},
                                  /*scan_output_arity*/ returned_types.size(),
                                  /*table_filter_set*/ nullptr,
                                  /*partition_indices*/ {},
                                  budget_bytes,
                                  /*max_file_processed*/ 1);

  auto splits = drive_provider(provider);
  REQUIRE(splits.size() >= 2);  // must have flushed mid-file at least once

  // Each split's bytes should stay within budget + one row group of slop.
  std::size_t total_seen = 0;
  for (auto const& split : splits) {
    auto const bytes = total_uncompressed_bytes(*split);
    INFO("intra-file split bytes=" << bytes << " budget=" << budget_bytes
                                   << " per_rg=" << per_rg_bytes);
    REQUIRE(bytes <= budget_bytes + per_rg_bytes);
    total_seen += bytes;
  }

  // No row groups dropped or duplicated.
  REQUIRE(total_seen == whole_file_bytes);

  std::filesystem::remove_all(dir);
}
