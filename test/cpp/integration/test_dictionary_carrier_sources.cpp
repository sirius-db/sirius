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
 * @brief enable_dictionary_carrier on Parquet, DuckDB-native, and compressed pinned scans.
 *
 * The planner carries a string column as DICTIONARY32 when its storage dictionary-encodes it and
 * the scan does not filter it, and every query over such a carrier returns DuckDB's results. A
 * compressed pin qualifies on either tier.
 */

#include "utils/gpu_execution_fixture.hpp"

#include <catch.hpp>
#include <duckdb/execution/column_binding_resolver.hpp>
#include <duckdb/planner/logical_operator.hpp>
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
    // `low`, `group` and `nullable` repeat a few values; `distinct_values` never repeats, so
    // neither storage keeps it as a dictionary.
    run_ok(
      "CREATE TABLE t AS SELECT range AS x, 'k' || (range % 10) AS low, 'g' || (range % 7) AS "
      "\"group\", CASE WHEN range % 5 = 0 THEN NULL ELSE 'n' || (range % 3) END AS nullable, "
      "'u' || range AS distinct_values FROM range(100000);");
    run_ok("CREATE TABLE dim AS SELECT 'k' || range AS key, range AS w FROM range(5);");
    run_ok("CHECKPOINT;");
    parquet = "read_parquet('" + (dir / "t.parquet").string() + "')";
    run_ok("COPY t TO '" + (dir / "t.parquet").string() + "' (FORMAT parquet);");
    run_ok("SET enable_dictionary_carrier = true;");
  }

  ~DictionaryCarrierFixture()
  {
    if (pinned) { con->Query("CALL unpin_table('t');"); }
    for (auto const* setting : {"enable_dictionary_carrier",
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
      plan << "input -> identity\n"
              "---\n"
              "input -> dictionary -> keys_offsets, keys_chars, indices\n"
              "---\n"
              "input -> dictionary -> keys_offsets, keys_chars, indices\n"
              "---\n"
              "input -> dictionary -> keys_offsets, keys_chars, indices\n"
              "---\n"
              "input -> identity\n";
    }
    run_ok("SET pin_table_compression = true;");
    run_ok("SET pin_table_compression_min_batch_size_bytes = 0;");
    run_ok("SET pin_table_compression_max_compressed_fraction = 2.0;");
    run_ok("SET pin_table_input_compression_plan_dir = '" + plan_dir.string() + "';");
    run_ok("CALL pin_table(format='duckdb', name='t', tier='" + tier + "');");
    pinned = true;
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
      CHECK(scan->sidecar_from_source_dictionaries == (!pinned && carried > 0));
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
         }) {
      std::string sql = query;
      sql.replace(sql.find("{}"), 2, source);
      compare_gpu_vs_cpu(sql);
    }
    compare_gpu_vs_cpu_ordered("SELECT low, sum(x) FROM " + source +
                               " GROUP BY low HAVING sum(x) > 0 ORDER BY low");
  }

  fs::path dir;
  std::string parquet;
  bool pinned = false;
};

}  // namespace

TEST_CASE_METHOD(DictionaryCarrierFixture,
                 "dictionary carrier - an unfiltered Parquet scan carries its dictionary columns",
                 "[integration][gpu_execution][dictionary_carrier]")
{
  CHECK(carried_dictionaries("SELECT low, count(*) FROM " + parquet + " GROUP BY low") == 1);
  CHECK(carried_dictionaries("SELECT low, distinct_values FROM " + parquet) == 1);
  CHECK(carried_dictionaries("SELECT low, nullable, \"group\" FROM " + parquet) == 3);
  // The reader's row filter would turn off cuDF's dictionary transcode.
  CHECK(carried_dictionaries("SELECT low, count(*) FROM " + parquet +
                             " WHERE x > 10 GROUP BY low") == 0);
  run_ok("SET enable_dictionary_carrier = false;");
  CHECK(carried_dictionaries("SELECT low, count(*) FROM " + parquet + " GROUP BY low") == 0);
}

TEST_CASE_METHOD(DictionaryCarrierFixture,
                 "dictionary carrier - a DuckDB-native scan carries its low-cardinality strings",
                 "[integration][gpu_execution][dictionary_carrier]")
{
  CHECK(carried_dictionaries("SELECT low, count(*) FROM t GROUP BY low") == 1);
  CHECK(carried_dictionaries("SELECT low, distinct_values FROM t") == 1);
  // A filtered column keeps its strings; the other carried column does not.
  CHECK(carried_dictionaries("SELECT low, \"group\" FROM t WHERE low > 'k1'") == 1);
  CHECK(carried_dictionaries("SELECT low, x FROM t WHERE x > 10") == 1);
  run_ok("SET enable_dictionary_carrier = false;");
  CHECK(carried_dictionaries("SELECT low, count(*) FROM t GROUP BY low") == 0);
}

TEST_CASE_METHOD(
  DictionaryCarrierFixture,
  "dictionary carrier - a compressed pin carries its dictionary columns on either tier",
  "[integration][gpu_execution][dictionary_carrier]")
{
  auto const tier = GENERATE(as<std::string>{}, "gpu", "host");
  CAPTURE(tier);
  pin_compressed(tier);
  CHECK(carried_dictionaries("SELECT low, count(*) FROM t GROUP BY low") == 1);
  CHECK(carried_dictionaries("SELECT low, nullable, \"group\" FROM t") == 3);
  // An identity plan stores `distinct_values` as plain strings.
  CHECK(carried_dictionaries("SELECT low, distinct_values FROM t") == 1);
  compare_all("t");
}

TEST_CASE_METHOD(DictionaryCarrierFixture,
                 "dictionary carrier - queries over Parquet carriers match DuckDB",
                 "[integration][gpu_execution][dictionary_carrier]")
{
  compare_all(parquet);
}

TEST_CASE_METHOD(DictionaryCarrierFixture,
                 "dictionary carrier - queries over DuckDB-native carriers match DuckDB",
                 "[integration][gpu_execution][dictionary_carrier]")
{
  compare_all("t");
}
