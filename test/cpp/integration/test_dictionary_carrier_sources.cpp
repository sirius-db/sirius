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
 * @file test_dictionary_carrier_sources.cpp
 * @brief enable_dictionary_carrier on compressed pinned scans.
 *
 * The planner carries a string column as DICTIONARY32 when its Simpatico pin dictionary-encodes it
 * with keys under the carrier cap and the scan does not filter it, and every query over such a
 * carrier returns DuckDB's results. A compressed pin qualifies on either tier.
 */

#include "utils/gpu_execution_fixture.hpp"
#include "utils/transparent_execution_test_utils.hpp"

#include <catch.hpp>
#include <duckdb/execution/column_binding_resolver.hpp>
#include <duckdb/planner/logical_operator.hpp>
#include <late_mat/defer_directive.hpp>
#include <op/sirius_physical_table_scan.hpp>
#include <planner/sirius_physical_plan_generator.hpp>
#include <unistd.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <string>

namespace {

namespace fs = std::filesystem;

// Plans up to the scan's physical schema, before the tier policy decides which carriers stay.
class carrier_planner : public sirius::planner::sirius_physical_plan_generator {
 public:
  using sirius_physical_plan_generator::sirius_physical_plan_generator;

  duckdb::unique_ptr<sirius::op::sirius_physical_operator> plan(
    duckdb::unique_ptr<duckdb::LogicalOperator> logical)
  {
    logical->ResolveOperatorTypes();
    duckdb::ColumnBindingResolver resolver;
    resolver.VisitOperator(*logical);
    return create_plan(*logical);
  }
};

sirius::op::sirius_physical_table_scan const* find_table_scan(
  sirius::op::sirius_physical_operator const& op)
{
  if (op.type == sirius::op::SiriusPhysicalOperatorType::TABLE_SCAN) {
    return &op.Cast<sirius::op::sirius_physical_table_scan>();
  }
  for (auto const& child : op.children) {
    if (auto const* found = find_table_scan(*child)) { return found; }
  }
  return nullptr;
}

class DictionaryCarrierFixture : public sirius::test::GpuExecutionFixture {
 public:
  DictionaryCarrierFixture()
  {
    static int counter = 0;
    dir = fs::temp_directory_path() / ("sirius_dictionary_carrier_" + std::to_string(::getpid()) +
                                       "_" + std::to_string(counter++));
    fs::create_directories(dir);
    // `low`, `group` and `nullable` repeat a few values; `distinct_values` never repeats; `wide`
    // repeats, but its keys take more than the carrier cap.
    run_ok(
      "CREATE TABLE t AS SELECT range AS x, 'k' || (range % 10) AS low, 'g' || (range % 7) AS "
      "\"group\", CASE WHEN range % 5 = 0 THEN NULL ELSE 'n' || (range % 3) END AS nullable, "
      "'u' || range AS distinct_values, 'w' || (range % 50000) || repeat('_', 40) AS wide FROM "
      "range(100000);");
    run_ok("CREATE TABLE dim AS SELECT 'k' || range AS key, range AS w FROM range(5);");
    run_ok("CHECKPOINT;");
    run_ok("SET enable_dictionary_carrier = true;");
  }

  ~DictionaryCarrierFixture()
  {
    if (pinned) { con->Query("CALL unpin_table('t');"); }
    if (pinned_mixed) { con->Query("CALL unpin_table('mixed');"); }
    for (auto const* setting : {"enable_dictionary_carrier",
                                "enable_duckdb_fallback",
                                "late_materialization_min_boundaries",
                                "late_materialization_min_value_x_boundaries",
                                "scan_task_batch_size",
                                "hash_partition_bytes",
                                "sirius_exchange_peers",
                                "pin_table_compression",
                                "pin_table_compression_min_batch_size_bytes",
                                "pin_table_compression_max_compressed_fraction",
                                "pin_table_input_compression_plan_dir"}) {
      con->Query("RESET " + std::string(setting) + ";");
    }
    std::error_code ignored;
    fs::remove_all(dir, ignored);
  }

  // Pin `t` Simpatico-compressed on @p tier, with dictionary plans for the strings that repeat.
  void pin_compressed(std::string const& tier)
  {
    auto const plan_dir = dir / "plans";
    fs::create_directories(plan_dir);
    {
      std::ofstream plan(plan_dir / "t.txt");
      // `x` is bitpacked and the codes of `group` too, so a filter on `x` decodes `group` on the
      // compacted dict_codes route.
      plan << "input -> bitpack -> chunk_min, chunk_count, chunk_bits, packed\n"
              "---\n"
              "input -> dictionary -> keys_offsets, keys_chars, indices\n"
              "---\n"
              "input -> dictionary -> keys_offsets, keys_chars, indices\n"
              "dictionary.indices -> bitpack -> chunk_min, chunk_count, chunk_bits, packed\n"
              "---\n"
              "input -> dictionary -> keys_offsets, keys_chars, indices\n"
              "---\n"
              "input -> identity\n"
              "---\n"
              "input -> dictionary -> keys_offsets, keys_chars, indices\n";
    }
    run_ok("SET pin_table_compression = true;");
    run_ok("SET pin_table_compression_min_batch_size_bytes = 0;");
    run_ok("SET pin_table_compression_max_compressed_fraction = 2.0;");
    run_ok("SET pin_table_input_compression_plan_dir = '" + plan_dir.string() + "';");
    run_ok("CALL pin_table(format='duckdb', name='t', tier='" + tier + "');");
    pinned = true;
  }

  // Pin `mixed` Simpatico-compressed on @p tier. Its first chunk's `m` keys fit the carrier cap,
  // later chunks' do not, so those chunks decode `m` to strings.
  void pin_mixed(std::string const& tier)
  {
    run_ok(
      "CREATE TABLE mixed AS SELECT range AS x, CASE WHEN range < 200000 THEN 'm' || (range % 10) "
      "ELSE 'x' || range || repeat('_', 40) END AS m, 'p' || range || repeat('_', 40) AS p FROM "
      "range(300000);");
    run_ok("CREATE TABLE probe AS SELECT 'm' || range AS key FROM range(5);");
    run_ok("CHECKPOINT;");
    auto const plan_dir = dir / "mixed_plans";
    fs::create_directories(plan_dir);
    {
      std::ofstream plan(plan_dir / "mixed.txt");
      plan << "input -> bitpack -> chunk_min, chunk_count, chunk_bits, packed\n"
              "---\n"
              "input -> dictionary -> keys_offsets, keys_chars, indices\n"
              "---\n"
              "input -> identity\n";
    }
    run_ok("SET pin_table_compression = true;");
    run_ok("SET pin_table_compression_min_batch_size_bytes = 0;");
    run_ok("SET pin_table_compression_max_compressed_fraction = 2.0;");
    run_ok("SET pin_table_input_compression_plan_dir = '" + plan_dir.string() + "';");
    run_ok("SET scan_task_batch_size = 4194304;");
    run_ok("CALL pin_table(format='duckdb', name='mixed', tier='" + tier + "');");
    pinned_mixed = true;
  }

  // The number of scan outputs the planner carries as DICTIONARY32.
  std::size_t carried_dictionaries(std::string const& query)
  {
    run_ok("SET gpu_execution = false;");
    run_ok("BEGIN TRANSACTION READ ONLY;");
    carrier_planner planner(*con->context);
    auto const plan  = planner.plan(con->ExtractPlan(query));
    auto const* scan = find_table_scan(*plan);
    REQUIRE(scan != nullptr);
    std::size_t carried = 0;
    if (scan->has_physical_overrides()) {
      carried = static_cast<std::size_t>(std::ranges::count_if(
        scan->get_physical_types(),
        [](cudf::data_type type) { return type.id() == cudf::type_id::DICTIONARY32; }));
    }
    run_ok("ROLLBACK;");
    run_ok("SET gpu_execution = true;");
    return carried;
  }

  void compare_all(std::string const& source)
  {
    for (auto const& query : {
           "SELECT low, count(*), count(low), sum(x) FROM {} GROUP BY low",
           "SELECT low, nullable, count(*), count(nullable), min(distinct_values) FROM {} GROUP "
           "BY ALL",
           "SELECT \"group\", low, max(x) FROM {} GROUP BY ALL",
           "SELECT count(DISTINCT low), count(DISTINCT nullable) FROM {}",
           "SELECT DISTINCT nullable FROM {}",
           "SELECT upper(low), count(*) FROM {} GROUP BY 1",
           "SELECT low, count(*) FROM {} WHERE x % 3 = 0 GROUP BY low",
           "SELECT low, sum(w) FROM {} JOIN dim ON low = key GROUP BY low",
           "SELECT \"group\", count(*), sum(x) FROM {} WHERE x < 30000 GROUP BY \"group\"",
           "SELECT \"group\", low, count(*) FROM {} WHERE x BETWEEN 1000 AND 1999 GROUP BY ALL",
           "SELECT wide, count(*) FROM {} GROUP BY wide",
           "SELECT low, count(*) FROM {} WHERE x < 0 GROUP BY low",
         }) {
      std::string sql = query;
      sql.replace(sql.find("{}"), 2, source);
      compare_gpu_vs_cpu(sql);
    }
    compare_gpu_vs_cpu_ordered("SELECT low, sum(x) FROM " + source +
                               " GROUP BY low HAVING sum(x) > 0 ORDER BY low");
  }

  fs::path dir;
  bool pinned       = false;
  bool pinned_mixed = false;
};

}  // namespace

TEST_CASE_METHOD(
  DictionaryCarrierFixture,
  "dictionary carrier - a compressed pin carries its dictionary columns on either tier",
  "[integration][gpu_execution][dictionary_carrier]")
{
  auto const tier = GENERATE(as<std::string>{}, "gpu", "host");
  CAPTURE(tier);
  CHECK(carried_dictionaries("SELECT low, count(*) FROM t GROUP BY low") == 0);
  pin_compressed(tier);
  CHECK(carried_dictionaries("SELECT low, count(*) FROM t GROUP BY low") == 1);
  CHECK(carried_dictionaries("SELECT low, nullable, \"group\" FROM t") == 3);
  // An identity plan stores `distinct_values` as plain strings.
  CHECK(carried_dictionaries("SELECT low, distinct_values FROM t") == 1);
  CHECK(carried_dictionaries("SELECT wide, count(*) FROM t GROUP BY wide") == 0);
  compare_all("t");

  // The carrier reaches the scan's output batches, which the planner hands to the aggregate.
  auto const before = sirius::test::get_compressed_materialization_stats(*con);
  compare_gpu_vs_cpu("SELECT low, count(*) FROM t GROUP BY low");
  auto const after = sirius::test::get_compressed_materialization_stats(*con);
  CHECK(after.scan_dictionary_carriers > before.scan_dictionary_carriers);
}

TEST_CASE_METHOD(DictionaryCarrierFixture,
                 "dictionary carrier - deferred columns keep a carrier on either tier",
                 "[integration][gpu_execution][dictionary_carrier][late_mat]")
{
  auto const tier = GENERATE(as<std::string>{}, "gpu", "host");
  CAPTURE(tier);
  pin_compressed(tier);
  run_ok("SET enable_duckdb_fallback = false;");
  run_ok("SET late_materialization_min_boundaries = 0;");
  run_ok("SET late_materialization_min_value_x_boundaries = 0;");
  auto const before = sirius::late_mat::deferrals_installed();
  for (auto const* query : {
         "SELECT low, count(*) FROM t GROUP BY low",
         "SELECT low, distinct_values FROM t JOIN dim ON low = key WHERE x % 1000 = 0",
         "SELECT low, wide, distinct_values FROM t JOIN dim ON low = key WHERE x % 997 = 0",
         "SELECT \"group\", nullable, min(distinct_values) FROM t JOIN dim ON low = key GROUP BY "
         "ALL",
       }) {
    compare_gpu_vs_cpu(query);
  }
  // The host-tier pin installs no deferral for these queries.
  if (tier == "gpu") { CHECK(sirius::late_mat::deferrals_installed() > before); }
}

TEST_CASE_METHOD(DictionaryCarrierFixture,
                 "dictionary carrier - a column carried in some chunks and strings in others",
                 "[integration][gpu_execution][dictionary_carrier]")
{
  auto const tier = GENERATE(as<std::string>{}, "gpu", "host");
  CAPTURE(tier);
  pin_mixed(tier);
  REQUIRE(carried_dictionaries("SELECT m, count(*) FROM mixed GROUP BY m") == 1);
  run_ok("SET enable_duckdb_fallback = false;");
  run_ok("SET late_materialization_min_boundaries = 0;");
  run_ok("SET late_materialization_min_value_x_boundaries = 0;");
  compare_gpu_vs_cpu("SELECT m, count(*) FROM mixed GROUP BY m");
  compare_gpu_vs_cpu("SELECT m, p FROM mixed JOIN probe ON m = key");
  compare_gpu_vs_cpu("SELECT m, count(*), max(p) FROM mixed JOIN probe ON m = key GROUP BY m");
  compare_gpu_vs_cpu("SELECT DISTINCT m FROM mixed");
}

TEST_CASE_METHOD(DictionaryCarrierFixture,
                 "dictionary carrier - min and max of carried columns, split into partitions",
                 "[integration][gpu_execution][dictionary_carrier]")
{
  auto const tier = GENERATE(as<std::string>{}, "gpu", "host");
  CAPTURE(tier);
  pin_compressed(tier);
  REQUIRE(carried_dictionaries("SELECT low, min(nullable), max(nullable) FROM t GROUP BY low") > 0);
  run_ok("SET enable_duckdb_fallback = false;");
  for (bool const partitioned : {false, true}) {
    CAPTURE(partitioned);
    // About ten partitions of `t`, so aggregates merge and join feeders split their payloads.
    if (partitioned) { run_ok("SET hash_partition_bytes = 1048576;"); }
    for (auto const* query : {
           "SELECT min(low), max(low), min(nullable), max(nullable) FROM t",
           "SELECT low, min(nullable), max(nullable), count(*) FROM t GROUP BY low",
           "SELECT \"group\", min(low), max(low) FROM t GROUP BY \"group\"",
           "SELECT low, nullable, \"group\", w FROM t JOIN dim ON low = key WHERE x % 10 = 0",
           "SELECT nullable, count(*), sum(w) FROM t JOIN dim ON low = key GROUP BY nullable",
         }) {
      compare_gpu_vs_cpu(query);
    }
  }
}

TEST_CASE_METHOD(DictionaryCarrierFixture,
                 "dictionary carrier - ordered first and DISTINCT ON over carried columns",
                 "[integration][gpu_execution][dictionary_carrier][distinct]")
{
  auto const tier = GENERATE(as<std::string>{}, "gpu", "host");
  CAPTURE(tier);
  pin_compressed(tier);
  auto const* const keyed =
    "SELECT low, first(distinct_values ORDER BY x DESC) FROM t GROUP BY low";
  REQUIRE(carried_dictionaries(keyed) > 0);
  run_ok("SET enable_duckdb_fallback = false;");
  auto const before = sirius::test::get_compressed_materialization_stats(*con);
  compare_gpu_vs_cpu(keyed);
  CHECK(sirius::test::get_compressed_materialization_stats(*con).scan_dictionary_carriers >
        before.scan_dictionary_carriers);
  for (bool const partitioned : {false, true}) {
    CAPTURE(partitioned);
    if (partitioned) { run_ok("SET hash_partition_bytes = 1048576;"); }
    // Every order ends on the unique `x`, so each group has one winner to compare.
    for (auto const* query : {
           "SELECT low, first(nullable ORDER BY x) FROM t GROUP BY low",
           "SELECT \"group\", first(x ORDER BY low DESC, x) FROM t GROUP BY \"group\"",
           "SELECT \"group\", last(low ORDER BY x) FROM t GROUP BY \"group\"",
           "SELECT nullable, first(low ORDER BY x DESC) FROM t GROUP BY nullable",
           "SELECT DISTINCT ON (low) low, nullable, \"group\", x FROM t ORDER BY low, \"group\" "
           "DESC, x DESC",
           "SELECT low, first(nullable ORDER BY x) FROM t WHERE x < 0 GROUP BY low",
           "SELECT DISTINCT ON (low) low, \"group\" FROM t WHERE x < 0 ORDER BY low, x",
         }) {
      compare_gpu_vs_cpu(query);
    }
  }
}

TEST_CASE_METHOD(DictionaryCarrierFixture,
                 "dictionary carrier - ordered first over a key carried in some chunks only",
                 "[integration][gpu_execution][dictionary_carrier][distinct]")
{
  auto const tier = GENERATE(as<std::string>{}, "gpu", "host");
  CAPTURE(tier);
  pin_mixed(tier);
  REQUIRE(carried_dictionaries("SELECT m, first(p ORDER BY x DESC) FROM mixed GROUP BY m") == 1);
  run_ok("SET enable_duckdb_fallback = false;");
  compare_gpu_vs_cpu("SELECT m, first(p ORDER BY x DESC) FROM mixed GROUP BY m");
  compare_gpu_vs_cpu("SELECT m, last(x ORDER BY p) FROM mixed GROUP BY m");
  compare_gpu_vs_cpu("SELECT DISTINCT ON (m) m, p, x FROM mixed ORDER BY m, x DESC");
}

TEST_CASE_METHOD(DictionaryCarrierFixture,
                 "dictionary carrier - ordered first over deferred columns",
                 "[integration][gpu_execution][dictionary_carrier][distinct][late_mat]")
{
  auto const tier = GENERATE(as<std::string>{}, "gpu", "host");
  CAPTURE(tier);
  // pin_mixed checkpoints, which would make an earlier pin of `t` stale.
  pin_mixed(tier);
  pin_compressed(tier);
  run_ok("SET enable_duckdb_fallback = false;");
  run_ok("SET late_materialization_min_boundaries = 0;");
  run_ok("SET late_materialization_min_value_x_boundaries = 0;");
  // Only a group-by directly over the scan installs a deferral; a join or filter between stops it.
  auto const before = sirius::late_mat::deferrals_installed();
  for (auto const* query : {
         "SELECT low, first(distinct_values ORDER BY x DESC) FROM t GROUP BY low",
         "SELECT \"group\", first(nullable ORDER BY low DESC, x) FROM t GROUP BY \"group\"",
         "SELECT DISTINCT ON (low) low, nullable, distinct_values FROM t ORDER BY low, x DESC",
         "SELECT m, first(p ORDER BY x DESC) FROM mixed GROUP BY m",
         "SELECT DISTINCT ON (m) m, p FROM mixed ORDER BY m, x",
       }) {
    compare_gpu_vs_cpu(query);
  }
  // The host-tier pin installs no deferral for these queries.
  if (tier == "gpu") { CHECK(sirius::late_mat::deferrals_installed() > before); }
}

TEST_CASE_METHOD(DictionaryCarrierFixture,
                 "dictionary carrier - a distributed query plans none",
                 "[integration][gpu_execution][dictionary_carrier]")
{
  pin_compressed("gpu");
  REQUIRE(carried_dictionaries("SELECT low, count(*) FROM t GROUP BY low") == 1);
  run_ok("SET sirius_exchange_peers = '127.0.0.1:62998,127.0.0.1:62999';");
  CHECK(carried_dictionaries("SELECT low, count(*) FROM t GROUP BY low") == 0);
}
