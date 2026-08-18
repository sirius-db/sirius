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
 * @file test_pin_table_type_drift.cpp
 * @brief Verifies clean cache misses when a DuckDB pin's projected native types drift.
 *
 * A drop/recreate can preserve the qualified table name while changing column types. These tests
 * require the plan-time carrier probe and serve-time native-type gate to reject the stale pin on
 * both tiers. Fallback is disabled so an unrelated plan rejection cannot hide the result.
 */

#include <catch.hpp>
#include <duckdb.hpp>
#include <utils/gpu_execution_fixture.hpp>
#include <utils/transparent_execution_test_utils.hpp>

#include <string>

using PinDriftFixture = sirius::test::GpuExecutionFixture;

namespace {

/// One drop/recreate drift scenario over table `drift_t`.
struct drift_recipe {
  std::string pin_select;       ///< SELECT producing the pinned incarnation's rows
  std::string recreate_select;  ///< SELECT producing the recreated incarnation's rows
  std::string filter_query;     ///< selective count over the NEW values (old values miss it)
};

// Restore query settings and remove the pin during assertion unwinding. The destructor issues raw
// queries without assertions because throwing during unwinding would terminate the test process.
class drift_cleanup_guard {
 public:
  explicit drift_cleanup_guard(duckdb::Connection& con) : _con(con) {}
  drift_cleanup_guard(drift_cleanup_guard const&)            = delete;
  drift_cleanup_guard& operator=(drift_cleanup_guard const&) = delete;
  ~drift_cleanup_guard()
  {
    _con.Query("SET enable_duckdb_fallback = true;");
    _con.Query("SET enable_compressed_materialization = true;");
    _con.Query("CALL unpin_table('drift_t');");
  }

 private:
  duckdb::Connection& _con;
};

/// Pin `drift_t` (compressed materialization on, asserting the pin actually narrowed), drop and
/// recreate it per @p recipe, then verify GPU-only queries return the fresh values.
void run_drift_case(PinDriftFixture& fx,
                    std::string const& tier,
                    drift_recipe const& recipe,
                    bool compressed_materialization_at_query = true)
{
  drift_cleanup_guard cleanup{*fx.con};
  fx.run_ok("SET enable_compressed_materialization = true;");
  fx.run_ok("CREATE TABLE drift_t AS " + recipe.pin_select + ";");
  fx.run_ok("CHECKPOINT;");

  // The drift only bites when the stale chunks are physically narrower than the recreated type's
  // native carrier, so make the precondition explicit rather than trusting the value recipes.
  auto const pin_before = sirius::test::get_compressed_materialization_stats(*fx.con);
  fx.run_ok("CALL pin_table(format='duckdb', name='drift_t', tier='" + tier + "');");
  auto const pin_after = sirius::test::get_compressed_materialization_stats(*fx.con);
  REQUIRE(pin_after.pin_columns_narrowed > pin_before.pin_columns_narrowed);

  fx.run_ok("DROP TABLE drift_t;");
  fx.run_ok("CREATE TABLE drift_t AS " + recipe.recreate_select + ";");
  fx.run_ok("CHECKPOINT;");

  if (!compressed_materialization_at_query) {
    fx.run_ok("SET enable_compressed_materialization = false;");
  }
  fx.run_ok("SET enable_duckdb_fallback = false;");
  // count(a), not count(*): a zero-column scan requests the rowid sentinel, which an MVCC pin can
  // never serve, so the plan-time cache-or-CPU guard would reject it before the drift matters.
  fx.compare_gpu_vs_cpu("SELECT count(a) FROM drift_t;");
  fx.compare_gpu_vs_cpu_ordered("SELECT a FROM drift_t ORDER BY a LIMIT 5;");
  fx.compare_gpu_vs_cpu(recipe.filter_query);
}

/// No-drift positive control: pin `drift_t` narrowed, run the same GPU-only queries WITHOUT a
/// drop/recreate, and prove the pin still cache-serves. With compressed materialization off at
/// query time no narrow sidecar exists, so a scan column can only arrive narrower than its native
/// mapping -- and be RESTORED (int16 -> int32 at the scan boundary) -- when the cached narrowed
/// chunks served it; a disk-native fresh read decodes at native width and moves neither narrowing
/// counter. Without this control the drift cases above could pass by never serving any pin at all.
void run_no_drift_case(PinDriftFixture& fx, std::string const& tier)
{
  drift_cleanup_guard cleanup{*fx.con};
  fx.run_ok("SET enable_compressed_materialization = true;");
  fx.run_ok("CREATE TABLE drift_t AS SELECT (10000 + range)::INTEGER AS a FROM range(100);");
  fx.run_ok("CHECKPOINT;");

  auto const pin_before = sirius::test::get_compressed_materialization_stats(*fx.con);
  fx.run_ok("CALL pin_table(format='duckdb', name='drift_t', tier='" + tier + "');");
  auto const pin_after = sirius::test::get_compressed_materialization_stats(*fx.con);
  REQUIRE(pin_after.pin_columns_narrowed > pin_before.pin_columns_narrowed);

  fx.run_ok("SET enable_compressed_materialization = false;");
  fx.run_ok("SET enable_duckdb_fallback = false;");
  auto const before = sirius::test::get_compressed_materialization_stats(*fx.con);
  fx.compare_gpu_vs_cpu("SELECT count(a) FROM drift_t;");
  fx.compare_gpu_vs_cpu("SELECT count(*) FROM drift_t WHERE a >= 10050;");
  auto const after = sirius::test::get_compressed_materialization_stats(*fx.con);
  REQUIRE(after.scan_columns_restored > before.scan_columns_restored);
  REQUIRE(after.scan_columns_narrowed == before.scan_columns_narrowed);
}

}  // namespace

TEST_CASE_METHOD(PinDriftFixture,
                 "pin_table type drift - INTEGER pin recreated as DATE serves fresh dates",
                 "[integration][gpu_execution][pin_table_type_drift]")
{
  // Case A: the narrowed INT16 chunks (10000..10099) must NOT come back retagged as the epoch
  // days 10000..10099 (1997 dates).
  for (auto const* tier : {"gpu", "host"}) {
    DYNAMIC_SECTION("tier = " << tier)
    {
      run_drift_case(
        *this,
        tier,
        {.pin_select      = "SELECT (10000 + range)::INTEGER AS a FROM range(100)",
         .recreate_select = "SELECT DATE '2020-01-01' + range::INTEGER AS a FROM range(100)",
         .filter_query    = "SELECT count(*) FROM drift_t WHERE a >= DATE '2020-02-01';"});
    }
  }
}

TEST_CASE_METHOD(PinDriftFixture,
                 "pin_table type drift - DATE pin recreated as INTEGER serves fresh integers",
                 "[integration][gpu_execution][pin_table_type_drift]")
{
  // Case D: the narrowed DATE chunks (epoch days 18262..18361) must NOT come back as integers.
  for (auto const* tier : {"gpu", "host"}) {
    DYNAMIC_SECTION("tier = " << tier)
    {
      run_drift_case(
        *this,
        tier,
        {.pin_select      = "SELECT DATE '2020-01-01' + range::INTEGER AS a FROM range(100)",
         .recreate_select = "SELECT (5 + range)::INTEGER AS a FROM range(100)",
         .filter_query    = "SELECT count(*) FROM drift_t WHERE a >= 50;"});
    }
  }
}

TEST_CASE_METHOD(PinDriftFixture,
                 "pin_table type drift - narrowed pin misses cleanly with narrowing off at query "
                 "time",
                 "[integration][gpu_execution][pin_table_type_drift]")
{
  // Case E: with compressed materialization disabled at query time no sidecar exists, so the
  // serve-time identity gate is the only defense -- without it the resident INT16 chunks would
  // widen straight into TIMESTAMP_DAYS as retagged dates.
  for (auto const* tier : {"gpu", "host"}) {
    DYNAMIC_SECTION("tier = " << tier)
    {
      run_drift_case(
        *this,
        tier,
        {.pin_select      = "SELECT (10000 + range)::INTEGER AS a FROM range(100)",
         .recreate_select = "SELECT DATE '2020-01-01' + range::INTEGER AS a FROM range(100)",
         .filter_query    = "SELECT count(*) FROM drift_t WHERE a >= DATE '2020-02-01';"},
        /*compressed_materialization_at_query=*/false);
    }
  }
}

TEST_CASE_METHOD(PinDriftFixture,
                 "pin_table type drift - no-drift control: matching natives still cache-serve",
                 "[integration][gpu_execution][pin_table_type_drift]")
{
  // Positive control for the identity gates: with no drop/recreate the recorded pin-time natives
  // match the scan's, so the pin must keep serving (see run_no_drift_case for the stat used as
  // proof of serving).
  for (auto const* tier : {"gpu", "host"}) {
    DYNAMIC_SECTION("tier = " << tier) { run_no_drift_case(*this, tier); }
  }
}

TEST_CASE_METHOD(PinDriftFixture,
                 "pin_table type drift - BIGINT pin recreated as INTEGER serves fresh integers",
                 "[integration][gpu_execution][pin_table_type_drift]")
{
  // Case B: same-family integer drift. The stale INT8 chunks (1..100) restore losslessly to the
  // recreated column's INT32 carrier, so nothing downstream can tell them from real data -- only
  // the recorded pin-time native (INT64 vs INT32) catches this.
  for (auto const* tier : {"gpu", "host"}) {
    DYNAMIC_SECTION("tier = " << tier)
    {
      run_drift_case(*this,
                     tier,
                     {.pin_select      = "SELECT (1 + range)::BIGINT AS a FROM range(100)",
                      .recreate_select = "SELECT (1000 + range)::INTEGER AS a FROM range(100)",
                      .filter_query    = "SELECT count(*) FROM drift_t WHERE a >= 1050;"});
    }
  }
}
