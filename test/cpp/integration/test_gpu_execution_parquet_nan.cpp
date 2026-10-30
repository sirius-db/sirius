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
#include <utils/gpu_execution_fixture.hpp>
#include <utils/parquet_fixture_utils.hpp>

#include <atomic>
#include <functional>
#include <string>
#include <vector>

namespace {

// Observe the actual decoder boundary, not EXPLAIN text: a GPU query may still
// evaluate its entire predicate after reading. Restore the hook even on failure.
class ReaderFilterObservation {
 public:
  explicit ReaderFilterObservation(duckdb::Connection& con)
    : counters_(sirius::test::get_registered_sirius_context(con)->physical_counters()),
      previous_(std::move(counters_->parquet_filter_for_testing))
  {
    counters_->parquet_filter_for_testing = [this](bool filtered, uint64_t rows) {
      ++batches;
      if (filtered) ++filtered_batches;
      decoded_rows += rows;
    };
  }
  ~ReaderFilterObservation() { counters_->parquet_filter_for_testing = std::move(previous_); }
  std::atomic<uint64_t> batches{0}, filtered_batches{0}, decoded_rows{0};

 private:
  std::shared_ptr<sirius::op::scan::physical_check_counters> counters_;
  std::function<void(bool, uint64_t)> previous_;
};

class ParquetNaNFixture : public sirius::test::GpuExecutionFixture {
 public:
  ParquetNaNFixture()
  {
    run_ok("SET gpu_execution = false");
    run_ok("SET threads = 1");
    // Aligned 2048-row groups: mixed (256 NULL, 256 NaN), all NaN, all NULL.
    // The mixed group also includes both infinities, signed zero and finite
    // matches/nonmatches. Both physical floating widths share identical rows.
    run_ok(
      "CREATE TABLE nan_values AS SELECT i::INTEGER AS id, v::FLOAT AS f, v AS d, "
      "CASE WHEN i % 8 IN (0, 6) THEN NULL ELSE 0 END::DOUBLE AS lo, "
      "CASE WHEN i % 8 IN (1, 3) THEN NULL ELSE 1 END::DOUBLE AS hi "
      "FROM (SELECT i, CASE WHEN i >= 4096 THEN NULL "
      "WHEN i >= 2048 THEN 'NaN'::DOUBLE ELSE CASE i % 8 "
      "WHEN 0 THEN NULL WHEN 1 THEN 'NaN'::DOUBLE WHEN 2 THEN '-Infinity'::DOUBLE "
      "WHEN 3 THEN -1 WHEN 4 THEN CASE WHEN i % 16 = 4 THEN '-0.0'::DOUBLE ELSE 0 END "
      "WHEN 5 THEN 1 WHEN 6 THEN 2 ELSE 'Infinity'::DOUBLE END END AS v "
      "FROM range(6144) t(i))");
    run_ok("COPY (SELECT * FROM nan_values ORDER BY id) TO " + sirius::test::sql_literal(path()) +
           " (FORMAT PARQUET, ROW_GROUP_SIZE 2048)");
    auto layout = con->Query("SELECT row_group_num_rows, stats_null_count FROM parquet_metadata(" +
                             sirius::test::sql_literal(path()) +
                             ") WHERE path_in_schema = 'd' ORDER BY row_group_id");
    REQUIRE(layout);
    REQUIRE_FALSE(layout->HasError());
    REQUIRE(layout->RowCount() == 3);
    std::vector<int64_t> const nulls{256, 0, 2048};
    for (duckdb::idx_t i = 0; i < 3; ++i) {
      REQUIRE(layout->Collection().GetValue(0, i).GetValue<int64_t>() == 2048);
      REQUIRE(layout->Collection().GetValue(1, i).GetValue<int64_t>() == nulls[i]);
    }
    auto population = con->Query("SELECT (id // 2048), count(*) FILTER (WHERE isnan(d)) FROM " +
                                 scan() + " GROUP BY 1 ORDER BY 1");
    REQUIRE(population);
    REQUIRE_FALSE(population->HasError());
    REQUIRE(population->RowCount() == 3);
    std::vector<int64_t> const nans{256, 2048, 0};
    for (duckdb::idx_t i = 0; i < 3; ++i) {
      REQUIRE(population->Collection().GetValue(1, i).GetValue<int64_t>() == nans[i]);
    }
  }

  std::string path() const { return dir_.file("nan_predicates.parquet"); }
  std::string scan() const { return "read_parquet(" + sirius::test::sql_literal(path()) + ")"; }

  // Always compare row identities AND all projected values, with exact types,
  // one successful GPU execution and no plan-time/runtime CPU fallback.
  void check(std::string const& predicate, bool require_reader_filter = false)
  {
    INFO(predicate);
    ReaderFilterObservation observation(*con);
    auto before = sirius::test::get_transparent_execution_stats(*con);
    compare_gpu_vs_cpu("SELECT id, f, d, lo, hi FROM " + scan() + " WHERE " + predicate);
    auto after = sirius::test::get_transparent_execution_stats(*con);
    REQUIRE(observation.batches.load() > 0);
    if (require_reader_filter) {
      REQUIRE(after.parquet_reader_calls[path()] > before.parquet_reader_calls[path()]);
      REQUIRE(observation.filtered_batches.load() > 0);
      // A selective predicate must remove rows before the residual evaluator.
      REQUIRE(observation.decoded_rows.load() < 6144);
      REQUIRE(observation.decoded_rows.load() > 0);
    }
  }

 private:
  sirius::util::env_guard test_options_{"SIRIUS_ENABLE_TEST_OPTIONS", "1"};
  sirius::test::scratch_dir dir_{"parquet_nan"};
};

}  // namespace

TEST_CASE_METHOD(ParquetNaNFixture,
                 "Parquet floating comparisons preserve NaN ordering and NULLs end to end",
                 "[integration][gpu_execution][parquet][nan_predicates]")
{
  for (auto const* column : {"f", "d"}) {
    auto type = std::string(column) == "f" ? "FLOAT" : "DOUBLE";
    for (auto const* value : {"0", "'Infinity'", "'NaN'"}) {
      for (auto const* op : {"=", "<>", "<", "<=", ">", ">="}) {
        // >= NaN retains the all-NaN group; <= 0 and < NaN exercise selective
        // filtering inside the mixed group as well as whole-group decisions.
        // Equality may stay residual under the cuDF bloom-filter guard. An
        // empty > NaN result may be eliminated before the decoder runs.
        bool const reader =
          std::string(op) != "=" && !(std::string(op) == ">" && std::string(value) == "'NaN'");
        check(std::string(column) + " " + op + " " + value + "::" + type, reader);
      }
    }
    check(std::string(column) + " IS NULL");
    check(std::string(column) + " IS NOT NULL");
  }
}

TEST_CASE_METHOD(ParquetNaNFixture,
                 "Parquet floating BETWEEN and membership preserve NaN and NULL semantics",
                 "[integration][gpu_execution][parquet][nan_predicates]")
{
  for (auto const* column : {"f", "d"}) {
    auto type = std::string(column) == "f" ? "FLOAT" : "DOUBLE";
    auto nan  = std::string("'NaN'::") + type;
    for (auto const* negation : {"", "NOT "}) {
      auto prefix = std::string(column) + " " + negation;
      check(prefix + "BETWEEN 0::" + type + " AND " + nan, std::string(negation).empty());
      check(prefix + "BETWEEN " + nan + " AND " + nan);
      check(prefix + "BETWEEN lo::" + type + " AND hi::" + type);
      for (auto const& list :
           std::vector<std::string>{"0::" + std::string(type) + ", " + nan,
                                    "NULL::" + std::string(type) + ", " + nan,
                                    nan + ", NULL::" + type,
                                    "1::" + std::string(type) + ", NULL::" + type + ", " + nan}) {
        // DuckDB pushes this NOT IN as inequalities. Positive IN/equality,
        // NULL-bearing lists and NOT BETWEEN may remain residual predicates;
        // their full SQL results are still checked, without requiring a
        // planner rewrite or bypassing the reader's bloom-filter guard.
        bool const reader =
          std::string(negation) == "NOT " && list.find("NULL") == std::string::npos;
        check(prefix + "IN (" + list + ")", reader);
      }
    }
    // Project the predicates too: WHERE alone cannot distinguish FALSE from
    // UNKNOWN, especially NOT IN (NULL, NaN). Matching, nonmatching and NULL
    // probes must retain their values and validity, not only selected rows.
    ReaderFilterObservation observation(*con);
    compare_gpu_vs_cpu("SELECT id, " + std::string(column) + ", " + column + " BETWEEN lo::" +
                       type + " AND hi::" + type + ", " + column + " NOT BETWEEN lo::" + type +
                       " AND hi::" + type + ", " + column + " IN (NULL::" + type + ", " + nan +
                       "), " + column + " NOT IN (NULL::" + type + ", " + nan + ") FROM " + scan());
    REQUIRE(observation.batches.load() > 0);
    REQUIRE(observation.filtered_batches.load() == 0);
    REQUIRE(observation.decoded_rows.load() == 6144);
  }
}
