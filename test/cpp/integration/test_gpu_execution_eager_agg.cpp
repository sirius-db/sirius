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
 * @file test_gpu_execution_eager_agg.cpp
 * @brief End-to-end GPU-vs-CPU correctness for the eager-aggregation-pushdown
 *        pass (src/planner/eager_agg_pushdown_plan_pass.cpp) on the transparent
 *        execution path — the path where the rewrite actually runs in
 *        production. Each "fires" case additionally asserts the pass really
 *        fired (via its applied counter), so a silently-refused rewrite cannot
 *        make these tests vacuous; each "refused" case asserts the counter did
 *        NOT move and the results are still correct.
 *
 * The data is built so every NULL/no-match edge of the rewrite is exercised:
 * customers with zero orders (COUNT must be 0, not NULL, under LEFT/RIGHT
 * joins), NULL join keys on the pushed side (rows that never match), NULLs in
 * the aggregated column (COUNT skips them, SUM/MIN/MAX ignore them), duplicate
 * keys on BOTH sides (N:M multiplicity), and an empty pushed side.
 */

#include "log/level.hpp"
#include "log/sink.hpp"
#include "planner/eager_agg_pushdown_plan_pass.hpp"
#include "sirius_context.hpp"

#include <catch.hpp>
#include <duckdb.hpp>
#include <utils/gpu_execution_fixture.hpp>

#include <algorithm>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

/// RAII Sirius setting override: the setters write into the shared operator
/// params, so a test must put the previous value back.
class scoped_setting {
 public:
  scoped_setting(duckdb::Connection& con, std::string name, const std::string& value)
    : _con(con), _name(std::move(name))
  {
    auto current = con.Query("SELECT current_setting('" + _name + "')");
    REQUIRE(current);
    REQUIRE_FALSE(current->HasError());
    _original  = current->GetValue(0, 0).ToString();
    auto apply = con.Query("SET " + _name + " = " + value);
    REQUIRE(apply);
    REQUIRE_FALSE(apply->HasError());
  }
  ~scoped_setting() { _con.Query("SET " + _name + " = " + _original); }
  scoped_setting(const scoped_setting&)            = delete;
  scoped_setting& operator=(const scoped_setting&) = delete;

 private:
  duckdb::Connection& _con;
  std::string _name;
  std::string _original;
};

/// Log sink that records every message, so a test can observe a code path that
/// has no other externally visible effect (here: the generator throwing away a
/// rewritten plan and replanning the original).
class capturing_sink : public sirius::log::sink {
 public:
  void set_level(sirius::log::level level) override
  {
    std::lock_guard<std::mutex> guard(_mutex);
    _level = level;
  }
  [[nodiscard]] bool should_log(sirius::log::level level) const override
  {
    std::lock_guard<std::mutex> guard(_mutex);
    return level >= _level;
  }
  void log(sirius::log::level level,
           const std::source_location& /*location*/,
           std::string_view message) override
  {
    if (!should_log(level)) { return; }
    std::lock_guard<std::mutex> guard(_mutex);
    _messages.emplace_back(message);
  }
  bool flush() override { return true; }

  [[nodiscard]] bool contains(std::string_view needle) const
  {
    std::lock_guard<std::mutex> guard(_mutex);
    return std::ranges::any_of(_messages, [needle](const std::string& message) {
      return message.find(needle) != std::string_view::npos;
    });
  }

 private:
  mutable std::mutex _mutex;
  sirius::log::level _level = sirius::log::level::info;
  std::vector<std::string> _messages;
};

/// RAII install of a capturing sink, restoring whatever sink the suite had.
class scoped_capturing_sink {
 public:
  scoped_capturing_sink() : _sink(std::make_shared<capturing_sink>())
  {
    sirius::log::set_sink(_sink);
  }
  ~scoped_capturing_sink() { sirius::log::set_sink(_previous); }
  scoped_capturing_sink(const scoped_capturing_sink&)            = delete;
  scoped_capturing_sink& operator=(const scoped_capturing_sink&) = delete;

  capturing_sink& operator*() const { return *_sink; }

 private:
  std::shared_ptr<sirius::log::sink> _previous = sirius::log::get_sink();
  std::shared_ptr<capturing_sink> _sink;
};

class EagerAggFixture : public sirius::test::GpuExecutionFixture {
 public:
  EagerAggFixture()
  {
    // cust: c_id 4 is duplicated (N:M multiplicity through the join); c_id 5
    // has no orders (LEFT/RIGHT no-match rows -> COUNT 0 / SUM NULL).
    run_ok("CREATE TABLE cust (c_id INTEGER, c_grp INTEGER);");
    run_ok("INSERT INTO cust VALUES (1, 1), (2, 0), (3, 1), (4, 0), (4, 0), (5, 1), (6, 0);");
    // ord: duplicate join keys (three rows for c_id 1), a NULL join key (never
    // matches), NULLs in o_val (COUNT counts non-NULLs only), negatives for
    // MIN/MAX, and o_grp for multi-key joins. o_cid spans cust's full c_id
    // domain [1, 6] on purpose: a narrower domain would let DuckDB's
    // statistics propagation derive a c_id range filter on the cust scan,
    // and the benefit gate (correctly) refuses non-bare preserved sides —
    // which would mask the organic INNER fire path this file exercises.
    run_ok("CREATE TABLE ord (o_cid INTEGER, o_grp INTEGER, o_key INTEGER, o_val INTEGER);");
    run_ok(
      "INSERT INTO ord VALUES "
      "(1, 1, 100, 10), (1, 1, 101, NULL), (1, 0, 102, -7), "
      "(2, 0, 103, 20), (2, 0, 104, 20), "
      "(3, 1, 105, NULL), "
      "(4, 0, 106, -1), (4, 1, 107, 42), "
      "(6, 0, 109, NULL), (6, 1, 110, -3), "
      "(NULL, 1, 108, 99);");
    // ordempty: an empty pushed side.
    run_ok("CREATE TABLE ordempty (o_cid INTEGER, o_key INTEGER);");
    run_ok("CHECKPOINT;");
  }

  /// Rewrites EXECUTED on this connection. The generator bumps this only after
  /// the rewritten plan has survived every planning stage, so a rewrite that was
  /// built and then discarded (the fail-closed retry) never registers — which is
  /// what keeps the assertions below from passing vacuously on the un-rewritten
  /// plan's results.
  [[nodiscard]] std::uint64_t applied_count() const
  {
    auto state = duckdb::get_sirius_connection_state(*con->context);
    REQUIRE(state);
    return state->eager_agg_pushdown_applied_count();
  }

  /// compare_gpu_vs_cpu, requiring that the pass fired during the GPU planning
  /// (the CPU run disables transparent execution and never plans on Sirius).
  void compare_fired(const std::string& query)
  {
    auto before = applied_count();
    compare_gpu_vs_cpu(query);
    CHECK(applied_count() > before);
  }

  /// compare_gpu_vs_cpu, requiring that the pass did NOT fire.
  void compare_refused(const std::string& query)
  {
    auto before = applied_count();
    compare_gpu_vs_cpu(query);
    CHECK(applied_count() == before);
  }
};

}  // namespace

//===----------------------------------------------------------------------===//
// Fired shapes: rewritten GPU plan must match the CPU results exactly
//===----------------------------------------------------------------------===//

TEST_CASE_METHOD(EagerAggFixture,
                 "gpu_execution eager agg pushdown - LEFT join grouped COUNT (q13 shape)",
                 "[integration][gpu_execution][eager_agg]")
{
  // Customer 5 has no orders: COUNT must be 0 (COALESCE repair), not NULL.
  // Customer 3's only order has o_val NULL: count(o_val) = 0 for a MATCHED row.
  compare_fired(
    "SELECT c_id, count(o_key), count(o_val) FROM cust LEFT JOIN ord ON c_id = o_cid "
    "GROUP BY c_id");
}

TEST_CASE_METHOD(EagerAggFixture,
                 "gpu_execution eager agg pushdown - RIGHT join (DuckDB's flipped q13 plan)",
                 "[integration][gpu_execution][eager_agg]")
{
  compare_fired("SELECT c_id, count(o_key) FROM ord RIGHT JOIN cust ON o_cid = c_id GROUP BY c_id");
}

TEST_CASE_METHOD(EagerAggFixture,
                 "gpu_execution eager agg pushdown - INNER join COUNT and SUM",
                 "[integration][gpu_execution][eager_agg]")
{
  // COUNT over o_val (nullable), not o_key: under an INNER join DuckDB's
  // optimizer rewrites COUNT of a provably non-NULL column into count_star(),
  // which the matcher refuses by design (no argument to push). A nullable
  // argument keeps the plain count(col) shape this test exercises.
  compare_fired(
    "SELECT c_id, count(o_val), sum(o_val) FROM cust JOIN ord ON c_id = o_cid GROUP BY c_id");
}

TEST_CASE_METHOD(EagerAggFixture,
                 "gpu_execution eager agg pushdown - LEFT join SUM/MIN/MAX keep NULL semantics",
                 "[integration][gpu_execution][eager_agg]")
{
  // Unmatched customers must stay NULL for SUM/MIN/MAX (no COALESCE), and
  // customer 3 (only NULL o_val) must also be NULL.
  compare_fired(
    "SELECT c_id, sum(o_val), min(o_val), max(o_val) FROM cust LEFT JOIN ord ON c_id = o_cid "
    "GROUP BY c_id");
}

TEST_CASE_METHOD(EagerAggFixture,
                 "gpu_execution eager agg pushdown - multi-key equi join",
                 "[integration][gpu_execution][eager_agg]")
{
  compare_fired(
    "SELECT c_id, count(o_key) FROM cust LEFT JOIN ord ON c_id = o_cid AND c_grp = o_grp "
    "GROUP BY c_id");
}

TEST_CASE_METHOD(EagerAggFixture,
                 "gpu_execution eager agg pushdown - full q13 shape (second GROUP BY over counts)",
                 "[integration][gpu_execution][eager_agg]")
{
  compare_fired(
    "SELECT c_count, count(*) AS custdist FROM ("
    "  SELECT c_id, count(o_key) AS c_count FROM cust LEFT JOIN ord ON c_id = o_cid "
    "  GROUP BY c_id) GROUP BY c_count");
}

TEST_CASE_METHOD(EagerAggFixture,
                 "gpu_execution eager agg pushdown - empty pushed side yields COUNT 0 everywhere",
                 "[integration][gpu_execution][eager_agg]")
{
  compare_fired(
    "SELECT c_id, count(o_key) FROM cust LEFT JOIN ordempty ON c_id = o_cid GROUP BY c_id");
}

TEST_CASE_METHOD(EagerAggFixture,
                 "gpu_execution eager agg pushdown - forced pushdown on a filtered preserved side",
                 "[integration][gpu_execution][eager_agg]")
{
  // The default benefit gate refuses a filtered non-pushed side; FORCE bypasses
  // only the benefit heuristic, so the result must still be exact.
  scoped_setting force(*con, "eager_agg_pushdown_force", "true");
  compare_fired(
    "SELECT c_id, count(o_key) FROM cust LEFT JOIN ord ON c_id = o_cid WHERE c_grp = 1 "
    "GROUP BY c_id");
}

//===----------------------------------------------------------------------===//
// Refused shapes: pass must not fire, results must still be correct
//===----------------------------------------------------------------------===//

TEST_CASE_METHOD(EagerAggFixture,
                 "gpu_execution eager agg pushdown - refusals stay refused and correct",
                 "[integration][gpu_execution][eager_agg]")
{
  SECTION("count(*) counts join rows")
  {
    compare_refused("SELECT c_id, count(*) FROM cust LEFT JOIN ord ON c_id = o_cid GROUP BY c_id");
  }
  SECTION("avg is not decomposed")
  {
    compare_refused("SELECT c_id, avg(o_val) FROM cust JOIN ord ON c_id = o_cid GROUP BY c_id");
  }
  SECTION("DISTINCT aggregate")
  {
    compare_refused(
      "SELECT c_id, count(DISTINCT o_grp) FROM cust JOIN ord ON c_id = o_cid GROUP BY c_id");
  }
  SECTION("group key on the pushed side")
  {
    compare_refused("SELECT o_grp, count(o_key) FROM cust JOIN ord ON c_id = o_cid GROUP BY o_grp");
  }
  SECTION("aggregates over both sides")
  {
    compare_refused(
      "SELECT c_id, count(o_key), sum(c_grp) FROM cust JOIN ord ON c_id = o_cid GROUP BY c_id");
  }
  SECTION("filtered preserved side fails the default benefit gate")
  {
    compare_refused(
      "SELECT c_id, count(o_key) FROM cust LEFT JOIN ord ON c_id = o_cid WHERE c_grp = 1 "
      "GROUP BY c_id");
  }
}

TEST_CASE_METHOD(EagerAggFixture,
                 "gpu_execution eager agg pushdown - kill switch",
                 "[integration][gpu_execution][eager_agg]")
{
  scoped_setting off(*con, "enable_eager_agg_pushdown", "false");
  compare_refused(
    "SELECT c_id, count(o_key) FROM cust LEFT JOIN ord ON c_id = o_cid GROUP BY c_id");
}

//===----------------------------------------------------------------------===//
// Discarded rewrites: the fail-closed retry in create_plan
//===----------------------------------------------------------------------===//

TEST_CASE_METHOD(EagerAggFixture,
                 "gpu_execution eager agg pushdown - a rewrite that fails planning is discarded",
                 "[integration][gpu_execution][eager_agg]")
{
  // `ordblob` carries a BLOB column, which the duckdb-native GPU scan declines.
  // The matcher itself is happy with the shape (FORCE bypasses only the benefit
  // heuristic), so the generator builds the rewritten plan on its scratch
  // generator, watches create_plan_stages throw, and replans the untouched
  // original -- which the GPU planner declines for the same reason, so the query
  // lands on DuckDB CPU. Nothing about that sequence may leak: the counter must
  // not register the discarded rewrite, and the query must behave exactly as it
  // does with the pass switched off.
  run_ok("CREATE TABLE ordblob (o_cid INTEGER, o_key INTEGER, o_blob BLOB);");
  run_ok(
    "INSERT INTO ordblob VALUES "
    "(1, 200, 'aa'::BLOB), (1, 201, 'bb'::BLOB), (2, 202, NULL), (4, 203, 'cc'::BLOB);");
  run_ok("CHECKPOINT;");
  run_ok("SET gpu_execution = true;");

  const std::string query =
    "SELECT c_id, count(o_key) FROM cust LEFT JOIN ordblob ON c_id = o_cid GROUP BY c_id";

  // Reference answer from DuckDB with Sirius out of the picture entirely.
  std::vector<std::vector<std::string>> expected_rows;
  {
    scoped_setting cpu(*con, "gpu_execution", "false");
    auto result = con->Query(query);
    REQUIRE(result);
    REQUIRE_FALSE(result->HasError());
    expected_rows = collect_rows(*result);
  }

  // Baseline: the same query with the pass switched off, i.e. the exact plan the
  // retry is supposed to fall back to.
  auto const baseline_before = sirius::test::get_transparent_execution_stats(*con);
  std::vector<std::vector<std::string>> baseline_rows;
  {
    scoped_setting off(*con, "enable_eager_agg_pushdown", "false");
    auto result = con->Query(query);
    REQUIRE(result);
    REQUIRE_FALSE(result->HasError());
    baseline_rows = collect_rows(*result);
  }
  auto const baseline_after = sirius::test::get_transparent_execution_stats(*con);
  CHECK(baseline_rows == expected_rows);

  // Forced: the rewrite is built, fails planning, and is thrown away.
  auto const applied_before = applied_count();
  auto const forced_before  = sirius::test::get_transparent_execution_stats(*con);
  std::vector<std::vector<std::string>> forced_rows;
  bool retried = false;
  {
    scoped_setting force(*con, "eager_agg_pushdown_force", "true");
    scoped_capturing_sink sink;
    auto result = con->Query(query);
    REQUIRE(result);
    REQUIRE_FALSE(result->HasError());
    forced_rows = collect_rows(*result);
    retried     = (*sink).contains("retrying with the original plan");
  }
  auto const forced_after = sirius::test::get_transparent_execution_stats(*con);

  // The rewrite really was attempted -- otherwise the assertions below would
  // hold vacuously on a plan the pass never touched.
  CHECK(retried);
  // A discarded rewrite must not register as applied.
  CHECK(applied_count() == applied_before);
  CHECK(forced_rows == expected_rows);
  // Replanning the original is accounted for exactly once, same as with the pass
  // switched off: no double-counted rebind, fallback or execution.
  CHECK(forced_after.successful_rebinds - forced_before.successful_rebinds ==
        baseline_after.successful_rebinds - baseline_before.successful_rebinds);
  CHECK(forced_after.fallbacks - forced_before.fallbacks ==
        baseline_after.fallbacks - baseline_before.fallbacks);
  CHECK(forced_after.executions - forced_before.executions ==
        baseline_after.executions - baseline_before.executions);

  // The generator must be reusable afterwards: a shape the pass does fire on
  // still plans, runs on the GPU and registers.
  compare_fired("SELECT c_id, count(o_key) FROM cust LEFT JOIN ord ON c_id = o_cid GROUP BY c_id");
}
