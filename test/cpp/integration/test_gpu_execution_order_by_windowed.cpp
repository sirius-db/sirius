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

// ORDER BY through SQL with the windowed merge forced (`sort_merge_mode = 'windowed'`) and the
// smallest window and output chunk the settings accept, so every sort runs many merge rounds and
// many tasks per partition. Each query runs three times: on DuckDB's CPU execution, on the GPU
// with the windowed merge and on the GPU with `sort_merge_mode = 'one_shot'`. Both GPU results
// must match the CPU one.
//
// Tables are file-backed (GpuExecutionFixture) and scanned in small batches, so a sort partition
// holds many sorted runs.

#include <catch.hpp>
#include <op/sirius_physical_merge_sort.hpp>
#include <utils/gpu_execution_fixture.hpp>
#include <utils/scoped_sirius_setting.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

/// Compare every result column position by position: the query's order is total.
constexpr std::size_t kAllColumns = std::numeric_limits<std::size_t>::max();

/// What a query's sort is required to have done, read from the merge sort counters.
enum class merge_check {
  windowed,      ///< the windowed run merged at least one partition through windowed tasks
  intermediate,  ///< and some of those tasks were intermediate fan-in passes
  none,  ///< the plan has no merge sort, or the sort has no rows: only the result is checked
};

/// The rows of one result, each rendered as one string.
struct query_rows {
  std::vector<duckdb::LogicalType> types;
  /// Every column of each row, in emitted order.
  std::vector<std::string> rows;
  /// Length of the prefix of each row that renders its leading key columns.
  std::vector<std::size_t> key_lengths;

  [[nodiscard]] std::string_view key(std::size_t row) const
  {
    return std::string_view(rows[row]).substr(0, key_lengths[row]);
  }
};

class OrderByWindowedFixture : public sirius::test::GpuExecutionFixture {
 public:
  OrderByWindowedFixture()
  {
    // The smallest window and output chunk the settings accept: many rounds and many tasks.
    set("sort_merge_window_rows", "1024");
    set("sort_merge_output_bytes", "1048576");
    // Small scan batches: each one becomes a sorted run, so a partition merges many runs.
    set("scan_task_batch_size", "2097152");
    set("sort_merge_mode", "windowed");
  }

  ~OrderByWindowedFixture()
  {
    // Newest first, so a setting changed twice ends at the value it had before the fixture.
    while (!settings.empty()) {
      settings.pop_back();
    }
  }

  /// Sets a setting until the fixture is destroyed, which restores the value it had before.
  void set(const std::string& name, const std::string& value)
  {
    settings.push_back(std::make_unique<sirius::test::scoped_sirius_setting>(*con, name, value));
  }

  /// Set by a case whose fan-in cap is below its run count: its windowed runs must take
  /// intermediate passes.
  bool expect_intermediate_passes = false;

  /// Creates and persists a table, so the GPU native scan reads it from disk.
  void create_table(const std::string& name, const std::string& select)
  {
    run_ok("CREATE TABLE " + name + " AS " + select + ";");
    run_ok("CHECKPOINT;");
  }

  /**
   * @brief Requires the windowed and the one-shot GPU results of @p query to match DuckDB's.
   *
   * The leading @p key_columns result columns must be the query's sort keys: they are compared
   * position by position. The full rows are compared as multisets, since rows with equal keys may
   * come out in any order. With kAllColumns the query's order must be total, and the full rows
   * are compared position by position. @p check says which merge route the runs must have taken.
   */
  void check_order(const std::string& query,
                   std::size_t key_columns = kAllColumns,
                   merge_check check       = merge_check::windowed)
  {
    if (check == merge_check::windowed && expect_intermediate_passes) {
      check = merge_check::intermediate;
    }
    INFO(query);
    run_ok("SET gpu_execution = false;");
    auto cpu_result = con->Query(query);
    run_ok("SET gpu_execution = true;");
    REQUIRE(cpu_result);
    if (cpu_result->HasError()) { UNSCOPED_INFO("CPU error: " << cpu_result->GetError()); }
    REQUIRE_FALSE(cpu_result->HasError());
    auto cpu = collect(*cpu_result, key_columns);
    cpu_result.reset();

    auto windowed = run_on_gpu(query, "windowed", key_columns, check);
    auto one_shot = run_on_gpu(query, "one_shot", key_columns, check);

    require_same_keys(cpu, windowed, "windowed");
    require_same_keys(cpu, one_shot, "one_shot");
    if (key_columns < cpu.types.size()) {
      auto const cpu_hashes = sorted_row_hashes(cpu);
      require_same_rows(cpu, cpu_hashes, windowed, "windowed");
      require_same_rows(cpu, cpu_hashes, one_shot, "one_shot");
    }
  }

 private:
  /// Appends the rendering of row @p r of a flat @p column to @p out. The common column types
  /// are rendered from the vector's data, which is several times faster than through a Value.
  static void append_cell(const duckdb::DataChunk& chunk,
                          duckdb::idx_t c,
                          duckdb::idx_t r,
                          std::string& out)
  {
    auto const& column = chunk.data[c];
    if (!duckdb::FlatVector::Validity(column).RowIsValid(r)) {
      out += "NULL";
      return;
    }
    switch (column.GetType().id()) {
      case duckdb::LogicalTypeId::INTEGER:
        out += std::to_string(duckdb::FlatVector::GetData<int32_t>(column)[r]);
        break;
      case duckdb::LogicalTypeId::BIGINT:
        out += std::to_string(duckdb::FlatVector::GetData<int64_t>(column)[r]);
        break;
      case duckdb::LogicalTypeId::DOUBLE: {
        // The bit pattern: exact, and the same for every NaN the tables hold.
        std::uint64_t bits = 0;
        std::memcpy(&bits, &duckdb::FlatVector::GetData<double>(column)[r], sizeof(bits));
        out += std::to_string(bits);
        break;
      }
      case duckdb::LogicalTypeId::VARCHAR: {
        auto const& value = duckdb::FlatVector::GetData<duckdb::string_t>(column)[r];
        out.append(value.GetData(), value.GetSize());
        break;
      }
      default: out += chunk.GetValue(c, r).ToString(); break;
    }
  }

  static query_rows collect(duckdb::QueryResult& result, std::size_t key_columns)
  {
    query_rows out;
    out.types = result.GetTypes();
    out.rows.reserve(result.RowCount());
    out.key_lengths.reserve(result.RowCount());
    for (auto& chunk : result.Collection().Chunks()) {
      chunk.Flatten();
      for (duckdb::idx_t r = 0; r < chunk.size(); r++) {
        std::string row;
        std::size_t key_length = 0;
        for (duckdb::idx_t c = 0; c < chunk.ColumnCount(); c++) {
          append_cell(chunk, c, r, row);
          row += '\x1f';
          if (c < key_columns) { key_length = row.size(); }
        }
        out.rows.push_back(std::move(row));
        out.key_lengths.push_back(key_length);
      }
    }
    return out;
  }

  /// Runs @p query with the given sort_merge_mode and requires that it ran on the GPU.
  query_rows run_on_gpu(const std::string& query,
                        const std::string& mode,
                        std::size_t key_columns,
                        merge_check check)
  {
    INFO("sort_merge_mode = " << mode);
    run_ok("SET sort_merge_mode = '" + mode + "';");
    sirius::op::sirius_physical_merge_sort::reset_process_counters();
    auto const before = sirius::test::get_transparent_execution_stats(*con);
    auto result       = con->Query(query);
    auto const after  = sirius::test::get_transparent_execution_stats(*con);
    auto const merges = sirius::op::sirius_physical_merge_sort::process_counters();
    REQUIRE(result);
    if (result->HasError()) { UNSCOPED_INFO("GPU error: " << result->GetError()); }
    REQUIRE_FALSE(result->HasError());
    // Exactly one GPU execution and no fallback.
    sirius::test::require_transparent_execution_delta(before, after, 1, 0, 1);
    // The route the merge took: a query that silently merged one-shot under 'windowed' fails.
    if (mode == "one_shot") {
      REQUIRE(merges.windowed_partitions == 0);
      REQUIRE(merges.windowed_tasks == 0);
      if (check != merge_check::none) { REQUIRE(merges.one_shot_partitions >= 1); }
    } else if (check != merge_check::none) {
      REQUIRE(merges.windowed_partitions >= 1);
      REQUIRE(merges.one_shot_partitions == 0);
      REQUIRE(merges.windowed_tasks >= merges.windowed_partitions);
      REQUIRE(merges.rounds.rows_out > 0);
      if (check == merge_check::intermediate) {
        REQUIRE(merges.intermediate_tasks >= 1);
      } else {
        REQUIRE(merges.intermediate_tasks == 0);
      }
    }
    return collect(*result, key_columns);
  }

  static void require_same_keys(const query_rows& cpu, const query_rows& gpu, const char* mode)
  {
    INFO("sort_merge_mode = " << mode);
    REQUIRE(gpu.types == cpu.types);
    REQUIRE(gpu.rows.size() == cpu.rows.size());
    for (std::size_t r = 0; r < cpu.rows.size(); r++) {
      if (gpu.key(r) != cpu.key(r)) {
        FAIL("row " << r << " is out of order: GPU=[" << gpu.rows[r] << "] CPU=[" << cpu.rows[r]
                    << "]");
      }
    }
  }

  static std::vector<std::size_t> sorted_row_hashes(const query_rows& result)
  {
    std::vector<std::size_t> hashes;
    hashes.reserve(result.rows.size());
    for (auto const& row : result.rows) {
      hashes.push_back(std::hash<std::string>{}(row));
    }
    std::sort(hashes.begin(), hashes.end());
    return hashes;
  }

  /// Requires both results to hold the same multiset of rows. The row hashes are compared first;
  /// the rows themselves are sorted only to report a difference.
  static void require_same_rows(const query_rows& cpu,
                                const std::vector<std::size_t>& cpu_hashes,
                                const query_rows& gpu,
                                const char* mode)
  {
    INFO("sort_merge_mode = " << mode);
    if (sorted_row_hashes(gpu) == cpu_hashes) { return; }
    auto cpu_rows = cpu.rows;
    auto gpu_rows = gpu.rows;
    std::sort(cpu_rows.begin(), cpu_rows.end());
    std::sort(gpu_rows.begin(), gpu_rows.end());
    for (std::size_t r = 0; r < cpu_rows.size(); r++) {
      if (gpu_rows[r] != cpu_rows[r]) {
        FAIL("the results hold different rows; first difference in sorted order: GPU=["
             << gpu_rows[r] << "] CPU=[" << cpu_rows[r] << "]");
      }
    }
    FAIL("the results hold different rows");
  }

  std::vector<std::unique_ptr<sirius::test::scoped_sirius_setting>> settings;
};

const char* const kNullOrders[] = {
  "ASC NULLS FIRST",
  "ASC NULLS LAST",
  "DESC NULLS FIRST",
  "DESC NULLS LAST",
};

}  // namespace

TEST_CASE_METHOD(OrderByWindowedFixture,
                 "windowed merge ORDER BY: single INTEGER and BIGINT key",
                 "[integration][gpu_execution][order_by_windowed]")
{
  // ki repeats (about 100k distinct values); kb is close to unique, both signs, in [-2^62, 2^62).
  create_table("ow_int",
               "SELECT CASE WHEN i % 13 = 0 THEN NULL "
               "            ELSE CAST(hash(i) % 100003 AS INTEGER) - 50000 END AS ki, "
               "       CASE WHEN i % 17 = 0 THEN NULL "
               "            ELSE CAST(hash(i + 7) >> 1 AS BIGINT) - 4611686018427387904 END AS kb, "
               "       CAST(i AS BIGINT) AS id "
               "FROM range(1000000) AS t(i)");

  // Every null placement on the INTEGER key; the BIGINT key is sorted both ways below.
  for (const auto* null_order : kNullOrders) {
    check_order(std::string("SELECT ki, id FROM ow_int ORDER BY ki ") + null_order, 1);
  }
  check_order("SELECT kb, id FROM ow_int ORDER BY kb ASC NULLS FIRST", 1);
  // The default null placement, with a total order.
  check_order("SELECT ki, id FROM ow_int ORDER BY ki, id");
  check_order("SELECT kb, ki, id FROM ow_int ORDER BY kb DESC, id DESC");
}

TEST_CASE_METHOD(OrderByWindowedFixture,
                 "windowed merge ORDER BY: multi-column keys with mixed directions",
                 "[integration][gpu_execution][order_by_windowed]")
{
  create_table("ow_multi",
               "SELECT CASE WHEN i % 23 = 0 THEN NULL ELSE CAST(hash(i) % 50 AS INTEGER) END AS a, "
               "       CASE WHEN i % 19 = 0 THEN NULL "
               "            ELSE 'k' || CAST(hash(i + 1) % 500 AS VARCHAR) END AS s, "
               "       CASE WHEN hash(i + 2) % 10 = 0 THEN 'NaN'::DOUBLE "
               "            WHEN hash(i + 2) % 10 = 1 THEN NULL "
               "            ELSE CAST(hash(i + 3) % 2000 AS DOUBLE) / 8 - 100 END AS d, "
               "       CAST(i AS BIGINT) AS id "
               "FROM range(600000) AS t(i)");

  check_order("SELECT a, s, d, id FROM ow_multi ORDER BY a ASC, s DESC, d ASC, id");
  check_order(
    "SELECT s, d, a, id FROM ow_multi "
    "ORDER BY s DESC NULLS FIRST, d DESC NULLS LAST, a ASC NULLS FIRST",
    3);
  check_order("SELECT d, a, s, id FROM ow_multi ORDER BY d ASC, a DESC", 2);
  check_order("SELECT d, id FROM ow_multi ORDER BY d DESC NULLS FIRST, id DESC");
  check_order("SELECT s, a, id FROM ow_multi ORDER BY s, a DESC NULLS LAST, id DESC");
}

TEST_CASE_METHOD(OrderByWindowedFixture,
                 "windowed merge ORDER BY: VARCHAR keys with long values",
                 "[integration][gpu_execution][order_by_windowed]")
{
  // Up to 300 repeats of one of three characters, then a number: long shared prefixes, lengths
  // from a few bytes to hundreds, a 4000-byte value every 5000th row, empty strings and NULLs.
  // A 1024-row window of these is cut by its byte budget, not its row count.
  create_table("ow_long",
               "SELECT CASE WHEN i % 29 = 0 THEN NULL "
               "            WHEN i % 31 = 0 THEN '' "
               "            WHEN i % 5000 = 1 THEN repeat('z', 4000) || CAST(i AS VARCHAR) "
               "            WHEN i % 37 = 0 THEN 'ñandú-' || CAST(hash(i) % 1000 AS VARCHAR) "
               "            ELSE repeat(chr(CAST(97 + hash(i) % 3 AS INTEGER)), "
               "                        CAST((hash(i) >> 8) % 300 AS INTEGER)) "
               "                 || CAST(hash(i) % 100000 AS VARCHAR) END AS s, "
               "       CAST(i AS BIGINT) AS id "
               "FROM range(200000) AS t(i)");

  check_order("SELECT s, id FROM ow_long ORDER BY s", 1);
  check_order("SELECT s, id FROM ow_long ORDER BY s DESC NULLS FIRST", 1);
  check_order("SELECT s, id FROM ow_long ORDER BY s ASC NULLS FIRST, id DESC");
  // The long key is not part of the output.
  check_order("SELECT id FROM ow_long ORDER BY s DESC, id");
}

TEST_CASE_METHOD(OrderByWindowedFixture,
                 "windowed merge ORDER BY: projection narrower than the input",
                 "[integration][gpu_execution][order_by_windowed]")
{
  // b is unique, so ORDER BY b is a total order even when b is not selected.
  create_table("ow_proj",
               "SELECT CAST(hash(i) % 1000 AS INTEGER) AS a, "
               "       CAST((i * 7919) % 1000003 AS BIGINT) AS b, "
               "       'c' || CAST(hash(i + 5) % 100000 AS VARCHAR) AS c, "
               "       CAST(i AS DOUBLE) / 3 AS d, "
               "       CAST(hash(i + 9) % 10 AS INTEGER) AS e "
               "FROM range(600000) AS t(i)");

  check_order("SELECT a, c FROM ow_proj ORDER BY b");
  check_order("SELECT c FROM ow_proj ORDER BY b DESC");
  // Output columns in a different order than the table's, key in the middle.
  check_order("SELECT d, b, a FROM ow_proj ORDER BY b");
  // Two keys that are not selected; b breaks the ties of e.
  check_order("SELECT c, a FROM ow_proj ORDER BY e DESC, b");
  // Ties on a key that is not selected: only the set of rows is defined.
  check_order("SELECT a, c FROM ow_proj ORDER BY e", 0);
}

TEST_CASE_METHOD(OrderByWindowedFixture,
                 "windowed merge ORDER BY: heavy ties, sorted and reverse-sorted input",
                 "[integration][gpu_execution][order_by_windowed]")
{
  create_table("ow_dist",
               "SELECT CAST(hash(i) % 5 AS INTEGER) AS few, "
               "       CAST(7 AS INTEGER) AS same, "
               "       CAST(i AS BIGINT) AS seq, "
               "       CAST(hash(i + 1) % 1000 AS INTEGER) AS pay "
               "FROM range(1000000) AS t(i)");

  // Few distinct values: every window holds long runs of equal keys.
  check_order("SELECT few, seq FROM ow_dist ORDER BY few", 1);
  check_order("SELECT few, seq, pay FROM ow_dist ORDER BY few DESC, seq");
  // One value in the whole column.
  check_order("SELECT same, seq FROM ow_dist ORDER BY same", 1);
  // Already sorted: the runs do not overlap.
  check_order("SELECT seq, pay FROM ow_dist ORDER BY seq");
  // Reverse sorted.
  check_order("SELECT seq, pay FROM ow_dist ORDER BY seq DESC");
}

TEST_CASE_METHOD(OrderByWindowedFixture,
                 "windowed merge ORDER BY: fan-in of 2 forces intermediate passes",
                 "[integration][gpu_execution][order_by_windowed]")
{
  set("sort_merge_max_fan_in", "2");
  expect_intermediate_passes = true;
  create_table("ow_fan",
               "SELECT CASE WHEN i % 11 = 0 THEN NULL "
               "            ELSE CAST(hash(i) % 20000 AS INTEGER) END AS k, "
               "       CASE WHEN i % 7 = 0 THEN NULL "
               "            ELSE 'v' || CAST(hash(i + 1) % 3000 AS VARCHAR) END AS s, "
               "       CASE WHEN hash(i + 2) % 50 = 0 THEN 'NaN'::DOUBLE "
               "            ELSE CAST(hash(i + 2) % 977 AS DOUBLE) / 4 END AS d, "
               "       CAST(i AS BIGINT) AS id "
               "FROM range(600000) AS t(i)");

  check_order("SELECT k, id FROM ow_fan ORDER BY k DESC NULLS FIRST", 1);
  check_order("SELECT k, s, d, id FROM ow_fan ORDER BY k, id");
  check_order("SELECT s, d, k, id FROM ow_fan ORDER BY s NULLS FIRST, d DESC", 2);
  // An intermediate pass rewrites every column, the final one emits a narrower projection.
  check_order("SELECT s FROM ow_fan ORDER BY d, id DESC");
}

TEST_CASE_METHOD(OrderByWindowedFixture,
                 "windowed merge ORDER BY: several sort partitions",
                 "[integration][gpu_execution][order_by_windowed]")
{
  // Small sort partitions: many partitions merge at once and their chunks must still come out in
  // partition order.
  set("max_sort_partition_bytes", "2097152");
  create_table("ow_part",
               "SELECT CASE WHEN i % 13 = 0 THEN NULL "
               "            ELSE CAST(hash(i) % 250000 AS INTEGER) END AS k, "
               "       'p' || CAST(hash(i + 1) % 1000 AS VARCHAR) AS s, "
               "       CAST(i AS BIGINT) AS id "
               "FROM range(1000000) AS t(i)");

  check_order("SELECT k, id FROM ow_part ORDER BY k", 1);
  check_order("SELECT k, s, id FROM ow_part ORDER BY k DESC NULLS FIRST, id");
  check_order("SELECT s, k, id FROM ow_part ORDER BY s DESC, k NULLS FIRST, id DESC");
}

TEST_CASE_METHOD(OrderByWindowedFixture,
                 "windowed merge ORDER BY: subquery, CTE and LIMIT above the sort",
                 "[integration][gpu_execution][order_by_windowed]")
{
  create_table("ow_sub",
               "SELECT CASE WHEN i % 13 = 0 THEN NULL "
               "            ELSE CAST(hash(i) % 100000 AS INTEGER) END AS k, "
               "       CAST(hash(i + 1) % 100 AS INTEGER) AS g, "
               "       CAST(i AS BIGINT) AS id "
               "FROM range(600000) AS t(i)");

  // ORDER BY in a subquery with no LIMIT.
  check_order("SELECT * FROM (SELECT k, id FROM ow_sub ORDER BY k, id) AS sub");
  check_order(
    "SELECT k + 1 AS k1, id FROM (SELECT k, id, g FROM ow_sub ORDER BY k DESC, id) AS sub");
  check_order(
    "SELECT k, id FROM (SELECT k, id, g FROM ow_sub ORDER BY k NULLS FIRST, id) AS sub WHERE g < "
    "50");
  // ORDER BY inside a CTE.
  check_order(
    "WITH sorted AS (SELECT k, g, id FROM ow_sub ORDER BY k DESC NULLS LAST, id DESC) "
    "SELECT k, id FROM sorted");
  check_order(
    "WITH sorted AS (SELECT k, g, id FROM ow_sub WHERE g <> 3 ORDER BY g, k, id) "
    "SELECT g, k, id FROM sorted");
  // A LIMIT above the sort: the rows kept are the first ones in sort order.
  check_order("SELECT k, id FROM ow_sub ORDER BY k, id LIMIT 400000");
  check_order("SELECT k, id FROM ow_sub ORDER BY k DESC, id LIMIT 200000 OFFSET 250000");
  check_order(
    "SELECT * FROM (SELECT k, id FROM ow_sub ORDER BY k NULLS FIRST, id DESC) AS sub "
    "LIMIT 300000");
  // A small LIMIT is planned as a top-n, with no merge sort.
  check_order("SELECT k, id FROM ow_sub ORDER BY k, id LIMIT 100", kAllColumns, merge_check::none);
}

TEST_CASE_METHOD(OrderByWindowedFixture,
                 "windowed merge ORDER BY: inputs smaller than one window",
                 "[integration][gpu_execution][order_by_windowed]")
{
  create_table("ow_small",
               "SELECT CASE WHEN i % 5 = 0 THEN NULL ELSE CAST(hash(i) % 40 AS INTEGER) END AS k, "
               "       'r' || CAST(i AS VARCHAR) AS s, "
               "       CAST(i AS BIGINT) AS id "
               "FROM range(3000) AS t(i)");

  check_order("SELECT k, s, id FROM ow_small ORDER BY k DESC NULLS FIRST, id");
  // Fewer rows than one window.
  check_order("SELECT k, s, id FROM ow_small WHERE id < 100 ORDER BY k, id DESC");
  check_order("SELECT k, s FROM ow_small WHERE id = 42 ORDER BY k");
  // No rows reach the sort.
  check_order(
    "SELECT k, s FROM ow_small WHERE id < 0 ORDER BY k, s", kAllColumns, merge_check::none);
}
