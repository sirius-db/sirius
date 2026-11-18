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

// `COPY (query) TO '<path>' (FORMAT sirius_discard [, check_order true])` through plain SQL. On
// the GPU the query's rows end in the discard sink, which drops each chunk as it arrives and
// returns their count. The tests require that the COPY ran on the GPU (transparent-execution
// counters, merge sort counters), that it returns the query's row count, that the sink's summary
// line reports no order violation, that the query's output is not accumulated on the device, and
// that DuckDB's CPU copy function returns the same count.
//
// `sirius_generate` has no CPU body, so the CPU case and the cases that compare both engines use
// a file-backed table (GpuExecutionFixture).

#include "sirius_context.hpp"

#include <catch.hpp>
#include <cucascade/memory/memory_space.hpp>
#include <cucascade/memory/reservation_aware_resource_adaptor.hpp>
#include <op/sirius_physical_merge_sort.hpp>
#include <unistd.h>
#include <utils/gpu_execution_fixture.hpp>
#include <utils/log_test_utils.hpp>
#include <utils/scoped_sirius_setting.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

/// Sort keys with ties and nulls in every column. For VARCHAR, p1 and p2 are both the key range
/// and the length range: `s` is one of seven 3-digit keys plus up to six random letters.
const std::string kSortColumns =
  "{name: 'k1', type: 'BIGINT', dist: 'few_distinct', p1: 40, null_fraction: 0.1}, "
  "{name: 'k2', type: 'INTEGER', dist: 'uniform', p1: -100, p2: 100, null_fraction: 0.05}, "
  "{name: 's', type: 'VARCHAR', dist: 'uniform', p1: 3, p2: 9, null_fraction: 0.1}";

constexpr std::uint64_t kBatches      = 8;
constexpr std::uint64_t kRowsPerBatch = 50000;
constexpr std::int64_t kRows          = static_cast<std::int64_t>(kBatches * kRowsPerBatch);

/// `sirius_generate(...)` call text. @p columns is the body of the `columns := [...]` list.
std::string generate_call(std::uint64_t num_batches,
                          std::uint64_t rows_per_batch,
                          std::uint64_t seed,
                          std::string const& columns)
{
  return "sirius_generate(num_batches := " + std::to_string(num_batches) +
         ", rows_per_batch := " + std::to_string(rows_per_batch) +
         ", seed := " + std::to_string(seed) + ", columns := [" + columns + "])";
}

const char* const kMergeModes[] = {"one_shot", "windowed"};

/// What the query under a COPY is required to have done, read from the merge sort counters.
enum class sort_check {
  merged,  ///< the query's ORDER BY merged at least one partition on the route of the mode
  none,    ///< the query has no merge sort, or the sort has no rows
};

/// The `[discard_sink] rows=… bytes=… order_violations=… order_checked=… ranges_checked=…` line
/// of one COPY.
struct sink_summary {
  std::uint64_t rows{0};
  std::uint64_t bytes{0};
  std::uint64_t order_violations{0};
  /// 1 when the sink compared the order of its input, 0 when the check was off.
  std::uint64_t order_checked{0};
  /// How many tasks' key ranges the sink compared. 0 when no input carried a position.
  std::uint64_t ranges_checked{0};
};

/// What one COPY on the GPU returned and logged.
struct discard_result {
  std::int64_t count{0};
  sink_summary summary;
  /// The planner logged that it left the order check off.
  bool order_check_disabled{false};
};

/// The order check of a merged ORDER BY compared key ranges. A windowed merge emits many chunks,
/// so the sink compares more than one range; a one-shot merge may emit a partition as one chunk.
void require_ranges_checked(const discard_result& copy, const std::string& mode)
{
  if (mode == "windowed") {
    CHECK(copy.summary.ranges_checked > 1);
  } else {
    CHECK(copy.summary.ranges_checked >= 1);
  }
}

/// Samples the device's allocated bytes from a thread and keeps the highest value seen above the
/// level at construction. The adaptor's own peak is never reset, so it cannot compare two queries
/// of one process.
class device_peak_sampler {
 public:
  explicit device_peak_sampler(const cucascade::memory::reservation_aware_resource_adaptor& adaptor)
    : _adaptor(adaptor), _baseline(adaptor.get_total_allocated_bytes()), _peak(_baseline)
  {
    _thread = std::thread([this] {
      while (!_stop.load(std::memory_order_relaxed)) {
        _peak = std::max(_peak, _adaptor.get_total_allocated_bytes());
        std::this_thread::sleep_for(std::chrono::microseconds(50));
      }
    });
  }

  ~device_peak_sampler() { stop(); }

  device_peak_sampler(const device_peak_sampler&)            = delete;
  device_peak_sampler& operator=(const device_peak_sampler&) = delete;

  /// Stops sampling and returns the peak above the level at construction.
  std::size_t stop()
  {
    if (_thread.joinable()) {
      _stop.store(true, std::memory_order_relaxed);
      _thread.join();
    }
    return _peak - _baseline;
  }

 private:
  const cucascade::memory::reservation_aware_resource_adaptor& _adaptor;
  std::size_t _baseline;
  std::size_t _peak;
  std::atomic<bool> _stop{false};
  std::thread _thread;
};

class CopyDiscardFixture : public sirius::test::GpuExecutionFixture {
 public:
  CopyDiscardFixture()
  {
    // The smallest window and output chunk the settings accept: a windowed merge takes many
    // rounds and tasks, so the sink receives many chunks.
    set("sort_merge_window_rows", "1024");
    set("sort_merge_output_bytes", "1048576");
    // Small scan batches: a table scan gives the sort many runs.
    set("scan_task_batch_size", "2097152");

    static std::atomic<unsigned> counter{0};
    target = (std::filesystem::temp_directory_path() /
              ("sirius_copy_discard_" + std::to_string(getpid()) + "_" +
               std::to_string(counter.fetch_add(1)) + ".out"))
               .string();
    REQUIRE_FALSE(std::filesystem::exists(target));
  }

  ~CopyDiscardFixture()
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

  /// Creates and persists a table, so the GPU native scan reads it from disk.
  void create_table(const std::string& name, const std::string& select)
  {
    run_ok("CREATE TABLE " + name + " AS " + select + ";");
    run_ok("CHECKPOINT;");
  }

  /// The COPY statement that discards the rows of @p query. The target is never created.
  [[nodiscard]] std::string discard_copy(const std::string& query,
                                         const std::string& options = "") const
  {
    return "COPY (" + query + ") TO '" + target + "' (FORMAT sirius_discard" + options + ")";
  }

  /// `SELECT count(*)` over @p query, run on the GPU without a fallback.
  std::int64_t count_on_gpu(const std::string& query)
  {
    auto const sql = "SELECT count(*) FROM (" + query + ") AS counted";
    INFO(sql);
    run_ok("SET gpu_execution = true;");
    auto const before = sirius::test::get_transparent_execution_stats(*con);
    auto result       = con->Query(sql);
    auto const after  = sirius::test::get_transparent_execution_stats(*con);
    REQUIRE(result);
    if (result->HasError()) { UNSCOPED_INFO("GPU error: " << result->GetError()); }
    REQUIRE_FALSE(result->HasError());
    sirius::test::require_transparent_execution_delta(before, after, 1, 0, 1);
    REQUIRE(result->RowCount() == 1);
    return result->Collection().GetValue(0, 0).GetValue<std::int64_t>();
  }

  /**
   * @brief Runs the discard COPY of @p query on the GPU with the given sort_merge_mode.
   *
   * Requires one GPU execution without a fallback, COPY's result shape (one BIGINT row), the
   * merge route @p check names, exactly one sink summary line whose row count is the returned
   * count, and that the target was not created.
   */
  discard_result copy_on_gpu(const std::string& query,
                             const std::string& mode,
                             const std::string& options = "",
                             sort_check check           = sort_check::merged)
  {
    auto const sql = discard_copy(query, options);
    INFO(sql);
    INFO("sort_merge_mode = " << mode);
    run_ok("SET gpu_execution = true;");
    run_ok("SET sort_merge_mode = '" + mode + "';");

    std::vector<sirius::test::recording_log_sink::record> records;
    duckdb::unique_ptr<duckdb::QueryResult> result;
    sirius::op::sirius_physical_merge_sort::reset_process_counters();
    auto const before = sirius::test::get_transparent_execution_stats(*con);
    {
      // The planner logs at DEBUG when it leaves the order check off.
      sirius::test::scoped_recording_log_sink logs("debug");
      result  = con->Query(sql);
      records = logs.records();
    }
    auto const after  = sirius::test::get_transparent_execution_stats(*con);
    auto const merges = sirius::op::sirius_physical_merge_sort::process_counters();
    REQUIRE(result);
    if (result->HasError()) { UNSCOPED_INFO("GPU error: " << result->GetError()); }
    REQUIRE_FALSE(result->HasError());
    // Exactly one GPU execution and no fallback: the CPU copy function did not run.
    sirius::test::require_transparent_execution_delta(before, after, 1, 0, 1);
    REQUIRE_FALSE(std::filesystem::exists(target));

    // The route the merge took. A sort that fell back to the CPU advances neither counter.
    if (check == sort_check::merged) {
      if (mode == "one_shot") {
        REQUIRE(merges.one_shot_partitions >= 1);
        REQUIRE(merges.windowed_partitions == 0);
        REQUIRE(merges.windowed_tasks == 0);
      } else {
        REQUIRE(merges.windowed_partitions >= 1);
        REQUIRE(merges.one_shot_partitions == 0);
        REQUIRE(merges.windowed_tasks >= merges.windowed_partitions);
        REQUIRE(merges.rounds.rows_out > 0);
      }
    }

    discard_result out;
    REQUIRE(result->GetTypes() == duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::BIGINT});
    REQUIRE(result->RowCount() == 1);
    out.count = result->Collection().GetValue(0, 0).GetValue<std::int64_t>();

    std::size_t summaries = 0;
    for (auto const& record : records) {
      if (record.message.rfind("sirius_discard: check_order disabled", 0) == 0) {
        out.order_check_disabled = true;
      }
      // The sink declares its one count column: no schema-mismatch line may name it.
      if (record.message.find("DISCARD_SINK") != std::string::npos &&
          record.message.find("mismatch") != std::string::npos) {
        FAIL(record.message);
      }
      if (record.message.rfind("[discard_sink] ", 0) != 0) { continue; }
      INFO(record.message);
      REQUIRE(std::sscanf(record.message.c_str(),
                          "[discard_sink] rows=%" SCNu64 " bytes=%" SCNu64
                          " order_violations=%" SCNu64 " order_checked=%" SCNu64
                          " ranges_checked=%" SCNu64,
                          &out.summary.rows,
                          &out.summary.bytes,
                          &out.summary.order_violations,
                          &out.summary.order_checked,
                          &out.summary.ranges_checked) == 5);
      summaries++;
    }
    // One sink per COPY, and its pipeline finished once.
    REQUIRE(summaries == 1);
    REQUIRE(out.summary.rows == static_cast<std::uint64_t>(out.count));
    if (out.count > 0) { REQUIRE(out.summary.bytes > 0); }
    return out;
  }

  /// Runs the discard COPY of @p query on DuckDB's CPU execution and returns its count.
  std::int64_t copy_on_cpu(const std::string& query, const std::string& options = "")
  {
    auto const sql = discard_copy(query, options);
    INFO(sql);
    run_ok("SET gpu_execution = false;");
    auto const before = sirius::test::get_transparent_execution_stats(*con);
    auto result       = con->Query(sql);
    auto const after  = sirius::test::get_transparent_execution_stats(*con);
    run_ok("SET gpu_execution = true;");
    REQUIRE(result);
    if (result->HasError()) { UNSCOPED_INFO("CPU error: " << result->GetError()); }
    REQUIRE_FALSE(result->HasError());
    // Nothing was handed to the GPU.
    sirius::test::require_transparent_execution_delta(before, after, 0, 0, 0);
    REQUIRE_FALSE(std::filesystem::exists(target));
    REQUIRE(result->GetTypes() == duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::BIGINT});
    REQUIRE(result->RowCount() == 1);
    return result->Collection().GetValue(0, 0).GetValue<std::int64_t>();
  }

  /// The resource adaptor of the first GPU: its allocated bytes are the device memory in use.
  const cucascade::memory::reservation_aware_resource_adaptor& device_adaptor()
  {
    auto sirius_ctx = sirius::test::get_registered_sirius_context(*con);
    auto spaces =
      sirius_ctx->get_memory_manager().get_memory_spaces_for_tier(cucascade::memory::Tier::GPU);
    REQUIRE_FALSE(spaces.empty());
    auto* space = const_cast<cucascade::memory::memory_space*>(spaces.front());
    auto* adaptor =
      space->get_memory_resource_as<cucascade::memory::reservation_aware_resource_adaptor>();
    REQUIRE(adaptor != nullptr);
    return *adaptor;
  }

  /// A path in the temp directory that does not exist; the discard format must not create it.
  std::string target;

 private:
  std::vector<std::unique_ptr<sirius::test::scoped_sirius_setting>> settings;
};

}  // namespace

TEST_CASE_METHOD(CopyDiscardFixture,
                 "sirius_discard COPY returns the row count of an ORDER BY over generated data",
                 "[integration][gpu_execution][copy_discard]")
{
  auto const source = generate_call(kBatches, kRowsPerBatch, 42, kSortColumns);
  REQUIRE(count_on_gpu("SELECT * FROM " + source) == kRows);

  const std::string queries[] = {
    // Single key.
    "SELECT k1, k2, s FROM " + source + " ORDER BY k1",
    "SELECT k2, s FROM " + source + " ORDER BY k2 DESC NULLS FIRST",
    // VARCHAR key.
    "SELECT s, k1 FROM " + source + " ORDER BY s",
    "SELECT s, k1 FROM " + source + " ORDER BY s DESC NULLS FIRST",
    // Multi-key, mixed directions and null placements.
    "SELECT k1, k2, s FROM " + source + " ORDER BY k1 DESC, s ASC NULLS FIRST, k2",
    "SELECT s, k2, k1 FROM " + source + " ORDER BY s NULLS FIRST, k2 DESC NULLS LAST, k1",
  };
  for (auto const& query : queries) {
    auto const expected = count_on_gpu(query);
    REQUIRE(expected == kRows);
    for (const auto* mode : kMergeModes) {
      auto const copy = copy_on_gpu(query, mode);
      CHECK(copy.count == expected);
      // No check was asked for.
      CHECK(copy.summary.order_checked == 0);
      CHECK(copy.summary.order_violations == 0);
    }
  }
}

TEST_CASE_METHOD(CopyDiscardFixture,
                 "sirius_discard COPY with check_order reports no order violation",
                 "[integration][gpu_execution][copy_discard]")
{
  auto const source = generate_call(kBatches, kRowsPerBatch, 7, kSortColumns);

  // Every key is a plain column of the output, so the planner turns the check on.
  const std::string queries[] = {
    "SELECT k1, k2, s FROM " + source + " ORDER BY k1",
    "SELECT k1, k2, s FROM " + source + " ORDER BY k1 DESC NULLS FIRST",
    "SELECT s, k1 FROM " + source + " ORDER BY s",
    "SELECT s, k1 FROM " + source + " ORDER BY s DESC NULLS LAST",
    "SELECT k2, k1, s FROM " + source + " ORDER BY k1 DESC, k2, s NULLS FIRST",
    "SELECT s, k2, k1 FROM " + source + " ORDER BY s DESC NULLS FIRST, k2 ASC NULLS FIRST, k1 DESC",
  };
  for (auto const& query : queries) {
    for (const auto* mode : kMergeModes) {
      auto const copy = copy_on_gpu(query, mode, ", check_order true");
      CHECK(copy.count == kRows);
      CHECK_FALSE(copy.order_check_disabled);
      // The check ran and compared key ranges: a sink whose inputs lost their positions would
      // report no violation without comparing anything.
      CHECK(copy.summary.order_checked == 1);
      require_ranges_checked(copy, mode);
      CHECK(copy.summary.order_violations == 0);
    }
  }

  // A key that the output drops: the count is still right and the check is left off.
  for (const auto* mode : kMergeModes) {
    auto const copy =
      copy_on_gpu("SELECT s FROM " + source + " ORDER BY k1, k2", mode, ", check_order true");
    CHECK(copy.count == kRows);
    CHECK(copy.order_check_disabled);
    CHECK(copy.summary.order_checked == 0);
    CHECK(copy.summary.order_violations == 0);
  }
}

TEST_CASE_METHOD(CopyDiscardFixture,
                 "sirius_discard COPY over a table matches the CPU copy function",
                 "[integration][gpu_execution][copy_discard]")
{
  create_table("cd_tab",
               "SELECT CASE WHEN i % 13 = 0 THEN NULL "
               "            ELSE CAST(hash(i) % 100003 AS INTEGER) - 50000 END AS k, "
               "       CASE WHEN i % 19 = 0 THEN NULL "
               "            ELSE 'k' || CAST(hash(i + 1) % 5000 AS VARCHAR) END AS s, "
               "       CAST(i AS BIGINT) AS id "
               "FROM range(600000) AS t(i)");

  struct table_case {
    std::string query;
    sort_check check;
  };
  const table_case cases[] = {
    {"SELECT k, id FROM cd_tab ORDER BY k", sort_check::merged},
    {"SELECT s, k, id FROM cd_tab ORDER BY s DESC NULLS FIRST, k, id", sort_check::merged},
    {"SELECT k, s, id FROM cd_tab WHERE id % 3 = 0 ORDER BY k DESC, id", sort_check::merged},
    // No ORDER BY: the sink follows the query's last operator.
    {"SELECT k, s FROM cd_tab WHERE id % 2 = 0", sort_check::none},
    // No rows: COPY returns 0.
    {"SELECT k, s FROM cd_tab WHERE id < 0 ORDER BY k, s", sort_check::none},
  };
  for (auto const& c : cases) {
    auto const expected = count_on_gpu(c.query);
    // gpu_execution = false: DuckDB's copy function, which discards too.
    CHECK(copy_on_cpu(c.query) == expected);
    CHECK(copy_on_cpu(c.query, ", check_order true") == expected);
    for (const auto* mode : kMergeModes) {
      CHECK(copy_on_gpu(c.query, mode, "", c.check).count == expected);
      auto const checked = copy_on_gpu(c.query, mode, ", check_order true", c.check);
      CHECK(checked.count == expected);
      if (c.check == sort_check::merged) {
        CHECK_FALSE(checked.order_check_disabled);
        CHECK(checked.summary.order_checked == 1);
        require_ranges_checked(checked, mode);
      }
      CHECK(checked.summary.order_violations == 0);
    }
  }
}

TEST_CASE_METHOD(CopyDiscardFixture,
                 "a COPY to another format stays on DuckDB",
                 "[integration][gpu_execution][copy_discard]")
{
  create_table("cd_csv", "SELECT CAST(i AS BIGINT) AS id FROM range(1000) AS t(i)");

  // Only the discard format is handed to the GPU: any other COPY writes its file on the CPU.
  run_ok("SET gpu_execution = true;");
  auto const before = sirius::test::get_transparent_execution_stats(*con);
  auto result =
    con->Query("COPY (SELECT id FROM cd_csv ORDER BY id) TO '" + target + "' (FORMAT csv)");
  auto const after = sirius::test::get_transparent_execution_stats(*con);
  REQUIRE(result);
  if (result->HasError()) { UNSCOPED_INFO("COPY error: " << result->GetError()); }
  REQUIRE_FALSE(result->HasError());
  sirius::test::require_transparent_execution_delta(before, after, 0, 0, 0);
  REQUIRE(result->RowCount() == 1);
  CHECK(result->Collection().GetValue(0, 0).GetValue<std::int64_t>() == 1000);
  REQUIRE(std::filesystem::exists(target));
  CHECK(std::filesystem::file_size(target) > 0);
  std::filesystem::remove(target);
}

TEST_CASE_METHOD(CopyDiscardFixture,
                 "a prepared sirius_discard COPY returns the count each time it is executed",
                 "[integration][gpu_execution][copy_discard]")
{
  create_table("cd_prepared", "SELECT CAST(i AS BIGINT) AS id FROM range(20000) AS t(i)");

  run_ok("SET gpu_execution = true;");
  run_ok("PREPARE cd_prepared_copy AS " +
         discard_copy("SELECT id FROM cd_prepared ORDER BY id DESC") + ";");
  for (int execution = 0; execution < 2; ++execution) {
    INFO("execution " << execution);
    auto const before = sirius::test::get_transparent_execution_stats(*con);
    auto result       = con->Query("EXECUTE cd_prepared_copy;");
    auto const after  = sirius::test::get_transparent_execution_stats(*con);
    REQUIRE(result);
    if (result->HasError()) { UNSCOPED_INFO("EXECUTE error: " << result->GetError()); }
    REQUIRE_FALSE(result->HasError());
    // The GPU ran it, with no fallback to DuckDB's CPU copy.
    sirius::test::require_transparent_execution_delta(before, after, 1, 0, 1);
    REQUIRE(result->GetTypes() == duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::BIGINT});
    REQUIRE(result->RowCount() == 1);
    CHECK(result->Collection().GetValue(0, 0).GetValue<std::int64_t>() == 20000);
    REQUIRE_FALSE(std::filesystem::exists(target));
  }
  run_ok("DEALLOCATE cd_prepared_copy;");
}

TEST_CASE_METHOD(
  CopyDiscardFixture,
  "a sirius_discard COPY that fails on the GPU is finished by DuckDB's copy function",
  "[integration][gpu_execution][copy_discard]")
{
  create_table("cd_fallback", "SELECT CAST(i AS BIGINT) AS id FROM range(20000) AS t(i)");

  run_ok("SET gpu_execution = true;");
  run_ok("SET enable_duckdb_fallback = true;");
  // Fails the GPU attempt at run time; the CPU plan kept for the replay is DuckDB's COPY.
  run_ok("SET sirius_test_inject_transparent_gpu_error = 'copy discard boom';");
  auto const before = sirius::test::get_transparent_execution_stats(*con);
  auto result       = con->Query(discard_copy("SELECT id FROM cd_fallback ORDER BY id DESC"));
  auto const after  = sirius::test::get_transparent_execution_stats(*con);
  run_ok("SET sirius_test_inject_transparent_gpu_error = '';");
  REQUIRE(result);
  if (result->HasError()) { UNSCOPED_INFO("COPY error: " << result->GetError()); }
  REQUIRE_FALSE(result->HasError());
  sirius::test::require_transparent_execution_delta(before, after, 1, 0, 1, 1);
  REQUIRE(result->GetTypes() == duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::BIGINT});
  REQUIRE(result->RowCount() == 1);
  CHECK(result->Collection().GetValue(0, 0).GetValue<std::int64_t>() == 20000);
  REQUIRE_FALSE(std::filesystem::exists(target));
}

TEST_CASE_METHOD(CopyDiscardFixture,
                 "sirius_discard COPY does not accumulate the query's output on the device",
                 "[integration][gpu_execution][copy_discard]")
{
  // The sort's input is one BIGINT column (16 MB); the projection above it emits 16 (256 MB).
  // The discard sink drops each output chunk as it arrives, so the device holds the sort's input
  // and the chunks in flight, never the output. The result collector moves each chunk to host
  // memory as it arrives, so the same query into the collector is the reference for a path that
  // does not accumulate on the device either.
  constexpr std::uint64_t kWideBatches      = 8;
  constexpr std::uint64_t kWideRowsPerBatch = 250000;
  constexpr std::int64_t kWideRows   = static_cast<std::int64_t>(kWideBatches * kWideRowsPerBatch);
  constexpr std::size_t kWideColumns = 16;
  constexpr std::size_t kOutputBytes =
    static_cast<std::size_t>(kWideRows) * kWideColumns * sizeof(std::int64_t);

  auto const source =
    generate_call(kWideBatches,
                  kWideRowsPerBatch,
                  11,
                  "{name: 'k', type: 'BIGINT', dist: 'uniform', p1: 0, p2: 1000000000}");
  std::string wide = "SELECT k";
  for (std::size_t c = 1; c < kWideColumns; c++) {
    wide += ", k + " + std::to_string(c) + " AS c" + std::to_string(c);
  }
  wide += " FROM (SELECT k FROM " + source + " ORDER BY k) AS sorted";

  // The windowed merge emits the sorted rows in chunks; a one-shot merge emits them whole.
  set("sort_merge_mode", "windowed");
  auto const& adaptor = device_adaptor();

  std::size_t select_peak = 0;
  {
    INFO(wide);
    run_ok("SET gpu_execution = true;");
    auto const before = sirius::test::get_transparent_execution_stats(*con);
    device_peak_sampler sampler(adaptor);
    auto result      = con->Query(wide);
    select_peak      = sampler.stop();
    auto const after = sirius::test::get_transparent_execution_stats(*con);
    REQUIRE(result);
    if (result->HasError()) { UNSCOPED_INFO("GPU error: " << result->GetError()); }
    REQUIRE_FALSE(result->HasError());
    sirius::test::require_transparent_execution_delta(before, after, 1, 0, 1);
    REQUIRE(result->RowCount() == static_cast<duckdb::idx_t>(kWideRows));
  }

  std::size_t copy_peak = 0;
  {
    device_peak_sampler sampler(adaptor);
    auto const copy = copy_on_gpu(wide, "windowed");
    copy_peak       = sampler.stop();
    REQUIRE(copy.count == kWideRows);
  }

  INFO("device peak above the level before each query: SELECT = "
       << select_peak << " bytes, COPY = " << copy_peak << " bytes; the output is " << kOutputBytes
       << " bytes");
  // The COPY never held even half of its output on the device.
  CHECK(copy_peak < kOutputBytes / 2);
  // The collector path's peak is reported above for comparison only: two runs' sampled peaks
  // differ with scheduling, so they are not asserted against each other.
}
