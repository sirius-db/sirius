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

#include "planner/dynamic_filter/build_key_domain.hpp"

#include <absl/cleanup/cleanup.h>
#include <catch.hpp>
#include <duckdb/planner/operator/logical_comparison_join.hpp>
#include <utils/dynamic_filter_test_utils.hpp>
#include <utils/gpu_execution_fixture.hpp>
#include <utils/parquet_fixture_utils.hpp>
#include <utils/scoped_sirius_setting.hpp>
#include <utils/transparent_execution_test_utils.hpp>

#include <algorithm>
#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

namespace {

using sirius::test::scoped_sirius_setting;
using sirius::test::sql_literal;

enum class build_keys : bool { distinct, duplicated };
enum class declaration : bool { none, unique_k };

/// Expected result rows and pin serving for every run; domain and gate counts apply only with
/// dynamic filters enabled.
struct expected_join {
  std::size_t rows;
  bool known_domain    = true;
  bool gated           = false;
  bool served_from_pin = true;
};

class DomainEvidenceFixture : public sirius::test::GpuExecutionFixture {
 public:
  DomainEvidenceFixture()
    : gpu_on(*con, "gpu_execution", true),
      native_keys(*con, "enable_compressed_materialization", false),
      zone_maps_off(*con, "enable_dynamic_zone_map_filter", false),
      shape(*con, "statistics_propagation,join_order,build_side_probe_side")
  {
    write("probe.parquet",
          "SELECT i::BIGINT AS p, ((i % 200) * 1000003 + 7)::BIGINT AS k "
          "FROM range(400) t(i)");
  }

  ~DomainEvidenceFixture() { con->Query("CALL unpin_table(" + sql_literal(pin_name()) + ");"); }

  void write(std::string const& file, std::string const& select)
  {
    scoped_sirius_setting cpu(*con, "gpu_execution", false);
    run_ok("COPY (" + select + ") TO " + directory.file_literal(file) + " (FORMAT PARQUET);");
  }

  void build(std::string const& file,
             build_keys keys = build_keys::distinct,
             int begin       = 0,
             int end         = 100)
  {
    auto const key = keys == build_keys::duplicated ? "(i % 10)" : "i";
    write(file,
          "SELECT " + std::string{key} + "::BIGINT AS payload, (" + key +
            " * 1000003 + 7)::BIGINT AS k, ((i % 5) * 1000003 + 7)::BIGINT AS other, "
            "i::INTEGER AS marker FROM range(" +
            std::to_string(begin) + ", " + std::to_string(end) + ") t(i)");
  }

  void pin(std::string const& source,
           std::string const& tier,
           declaration declared = declaration::unique_k)
  {
    run_ok("CALL pin_table(" + directory.file_literal(source) + ", tier=" + sql_literal(tier) +
           ", name=" + sql_literal(pin_name()) + ", cols=['marker', 'other', 'payload', 'k']" +
           (declared == declaration::unique_k ? ", unique_cols=['k']" : "") + ");");
  }

  [[nodiscard]] std::string query(std::string const& source,
                                  int selected,
                                  std::string const& key = "k") const
  {
    // Keep the filtered build on the right; source, pin and projected key ordinals differ.
    return "SELECT p.p, b.payload, b.k FROM read_parquet(" +
           directory.file_literal("probe.parquet") +
           ") p JOIN (SELECT other, payload, k FROM read_parquet(" + source + ") WHERE marker < " +
           std::to_string(selected) + ") b ON p.k = b." + key;
  }

  void check(std::string const& sql,
             expected_join const& expect,
             std::vector<std::string> const& build_files)
  {
    INFO(sql);
    duckdb::unique_ptr<duckdb::QueryResult> cpu;
    {
      scoped_sirius_setting cpu_only(*con, "gpu_execution", false);
      cpu = con->Query(sql);
    }
    REQUIRE(cpu);
    REQUIRE_FALSE(cpu->HasError());
    REQUIRE(cpu->RowCount() == expect.rows);
    auto const expected = sirius::test::collect_rows(*cpu);

    for (bool const enabled : {false, true}) {
      CAPTURE(enabled, expect.known_domain, expect.gated, expect.served_from_pin);
      scoped_sirius_setting filters(*con, "enable_dynamic_filter", enabled);
      auto const before     = sirius::test::get_dynamic_filter_stats_snapshot(*con);
      auto before_execution = sirius::test::get_transparent_execution_stats(*con);
      auto result           = con->Query(sql);
      REQUIRE(result);
      if (result->HasError()) { UNSCOPED_INFO(result->GetError()); }
      REQUIRE_FALSE(result->HasError());
      auto after_execution = sirius::test::get_transparent_execution_stats(*con);
      sirius::test::require_transparent_execution_delta(before_execution, after_execution, 1, 0, 1);
      REQUIRE(result->GetTypes() == cpu->GetTypes());
      REQUIRE(sirius::test::collect_rows(*result) == expected);
      auto const after = sirius::test::get_dynamic_filter_stats_snapshot(*con);

      REQUIRE(after.publications_failed == before.publications_failed);
      REQUIRE(after.keys_build_exceeded_domain == before.keys_build_exceeded_domain);
      REQUIRE(after.keys_with_known_domain ==
              before.keys_with_known_domain + (enabled && expect.known_domain ? 1 : 0));
      REQUIRE(after.keys_skipped_domain_gate ==
              before.keys_skipped_domain_gate + (enabled && expect.gated ? 1 : 0));
      if (enabled && !expect.gated) {
        REQUIRE(after.filters_pushed > before.filters_pushed);
      } else {
        REQUIRE(after.filters_pushed == before.filters_pushed);
      }

      auto const probe = directory.file("probe.parquet");
      REQUIRE(after_execution.parquet_reader_calls[probe] >
              before_execution.parquet_reader_calls[probe]);
      for (auto const& file : build_files) {
        auto const path = directory.file(file);
        if (expect.served_from_pin) {
          REQUIRE(after_execution.parquet_reader_calls[path] ==
                  before_execution.parquet_reader_calls[path]);
        } else {
          REQUIRE(after_execution.parquet_reader_calls[path] >
                  before_execution.parquet_reader_calls[path]);
        }
      }
    }
  }

  [[nodiscard]] std::string pin_name() const { return attach_alias + "_domain"; }

  sirius::op::dynamic_filter_domain_evidence evidence(std::string const& sql)
  {
    scoped_sirius_setting cpu(*con, "gpu_execution", false);
    run_ok("BEGIN TRANSACTION;");
    absl::Cleanup rollback = [this] { con->Query("ROLLBACK;"); };
    auto plan              = con->ExtractPlan(sql);
    REQUIRE(plan);
    auto* node = plan.get();
    while (node->type != duckdb::LogicalOperatorType::LOGICAL_COMPARISON_JOIN) {
      REQUIRE(node->children.size() == 1);
      node = node->children.front().get();
    }
    auto context = sirius::test::get_registered_sirius_context(*con);
    duckdb::SiriusContext::SlotGuard stable_registry(
      *context, *con->context, sirius::exec::query_admission::access::planning);
    auto const result = sirius::planner::build_key_domain_evidence_for(
      node->Cast<duckdb::LogicalComparisonJoin>(), *con->context, context->get_scan_manager());
    REQUIRE(result.size() == 1);
    return result.front();
  }

  sirius::test::scratch_dir directory{"domain_evidence"};

 private:
  scoped_sirius_setting gpu_on;
  scoped_sirius_setting native_keys;
  scoped_sirius_setting zone_maps_off;
  sirius::test::disabled_optimizers_guard shape;
};

}  // namespace

TEST_CASE_METHOD(DomainEvidenceFixture,
                 "pinned sparse domains gate by row count at both coverage thresholds",
                 "[integration][gpu_execution][dynamic_filter][domain_evidence]")
{
  auto const tier      = GENERATE(std::string{"gpu"}, std::string{"host"});
  auto const threshold = GENERATE(std::string{"0.9"}, std::string{"0.45"});
  auto const selected  = GENERATE(95, 60);
  CAPTURE(tier, threshold, selected);
  scoped_sirius_setting coverage(
    *con, "dynamic_filter_domain_coverage_threshold", std::string_view{threshold});
  build("build.parquet");
  pin("build.parquet", tier);
  check(query(directory.file_literal("build.parquet"), selected),
        {.rows  = static_cast<std::size_t>(2 * selected),
         .gated = selected == 95 || threshold == "0.45"},
        {"build.parquet"});
}

TEST_CASE_METHOD(DomainEvidenceFixture,
                 "a false pin uniqueness declaration preserves duplicate joins and DISTINCT",
                 "[integration][gpu_execution][dynamic_filter][domain_evidence]")
{
  auto const tier      = GENERATE(std::string{"gpu"}, std::string{"host"});
  auto const threshold = GENERATE(std::string{"0.9"}, std::string{"0.45"});
  auto const null_key  = GENERATE(false, true);
  CAPTURE(tier, threshold, null_key);
  scoped_sirius_setting coverage(
    *con, "dynamic_filter_domain_coverage_threshold", std::string_view{threshold});
  if (null_key) {
    write("duplicates.parquet",
          "SELECT (i % 10)::BIGINT AS payload, "
          "CASE WHEN i = 0 THEN NULL::BIGINT ELSE ((i % 10) * 1000003 + 7)::BIGINT END AS k, "
          "((i % 5) * 1000003 + 7)::BIGINT AS other, i::INTEGER AS marker FROM range(100) t(i)");
  } else {
    build("duplicates.parquet", build_keys::duplicated);
  }
  pin("duplicates.parquet", tier);
  auto const join = query(directory.file_literal("duplicates.parquet"), 95);
  check(join,
        {.rows = null_key ? std::size_t{188} : std::size_t{190}, .gated = true},
        {"duplicates.parquet"});
  check("SELECT DISTINCT p, payload, k FROM (" + join + ") joined",
        {.rows = 20, .gated = true},
        {"duplicates.parquet"});
}

TEST_CASE_METHOD(DomainEvidenceFixture,
                 "pinned domain hints belong to their source column",
                 "[integration][gpu_execution][dynamic_filter][domain_evidence]")
{
  auto const tier      = GENERATE(std::string{"gpu"}, std::string{"host"});
  auto const threshold = GENERATE(std::string{"0.9"}, std::string{"0.45"});
  CAPTURE(tier, threshold);
  scoped_sirius_setting coverage(
    *con, "dynamic_filter_domain_coverage_threshold", std::string_view{threshold});
  build("build.parquet");
  SECTION("a pin without a declaration cannot gate")
  {
    pin("build.parquet", tier, declaration::none);
    check(query(directory.file_literal("build.parquet"), 95), {.rows = 190}, {"build.parquet"});
  }
  SECTION("a declaration for k cannot gate the other source column")
  {
    pin("build.parquet", tier);
    check(query(directory.file_literal("build.parquet"), 95, "other"),
          {.rows = 190},
          {"build.parquet"});
  }
}

TEST_CASE_METHOD(DomainEvidenceFixture,
                 "bound domain evidence is copied and discarded after a pin replacement",
                 "[integration][gpu_execution][dynamic_filter][domain_evidence][build_key_domain]")
{
  using source    = sirius::op::dynamic_filter_domain_source;
  auto const tier = GENERATE(std::string{"gpu"}, std::string{"host"});
  CAPTURE(tier);
  build("first.parquet");
  build("replacement.parquet");
  pin("first.parquet", tier);
  auto const sql    = query(directory.file_literal("first.parquet"), 60);
  auto const copied = evidence(sql);
  REQUIRE(copied.cardinality == 100);
  REQUIRE(copied.source == source::pinned_parquet_exact);
  REQUIRE(copied.declared_unique);
  auto const other = evidence(query(directory.file_literal("first.parquet"), 60, "other"));
  CHECK(other.cardinality == 100);
  CHECK(other.source == source::pinned_parquet_exact);
  CHECK_FALSE(other.declared_unique);

  for (auto const* option : {", binary_as_string=true", ", union_by_name=true"}) {
    CAPTURE(option);
    auto const changed_reader =
      evidence(query(directory.file_literal("first.parquet") + option, 60));
    CHECK(changed_reader.cardinality == 0);
    CHECK(changed_reader.source == source::unknown);
    CHECK_FALSE(changed_reader.declared_unique);
  }

  pin("replacement.parquet", tier, declaration::none);
  auto const stale = evidence(sql);
  CHECK(stale.cardinality == 0);
  CHECK(stale.source == source::unknown);
  CHECK_FALSE(stale.declared_unique);
  CHECK(copied.cardinality == 100);
  CHECK(copied.source == source::pinned_parquet_exact);
  CHECK(copied.declared_unique);
  auto const replacement = evidence(query(directory.file_literal("replacement.parquet"), 60));
  CHECK(replacement.cardinality == 100);
  CHECK(replacement.source == source::pinned_parquet_exact);
  CHECK_FALSE(replacement.declared_unique);
}

TEST_CASE_METHOD(DomainEvidenceFixture,
                 "multiplying Parquet lineage declines coverage without changing join rows",
                 "[integration][gpu_execution][dynamic_filter][domain_evidence]")
{
  auto const tier      = GENERATE(std::string{"gpu"}, std::string{"host"});
  auto const threshold = GENERATE(std::string{"0.9"}, std::string{"0.45"});
  CAPTURE(tier, threshold);
  scoped_sirius_setting coverage(
    *con, "dynamic_filter_domain_coverage_threshold", std::string_view{threshold});
  build("build.parquet");
  write("multiplier.parquet", "SELECT i::BIGINT AS n FROM range(2) t(i)");
  pin("build.parquet", tier);
  auto const sql =
    "SELECT p.p, b.payload, b.k FROM read_parquet(" + directory.file_literal("probe.parquet") +
    ") p JOIN (SELECT x.payload, x.k FROM read_parquet(" + directory.file_literal("build.parquet") +
    ") x CROSS JOIN read_parquet(" + directory.file_literal("multiplier.parquet") +
    ") m WHERE x.marker < 95) b ON p.k = b.k";
  auto const domain = evidence(sql);
  REQUIRE(domain.cardinality == 0);
  REQUIRE(domain.source == sirius::op::dynamic_filter_domain_source::unknown);
  REQUIRE_FALSE(domain.declared_unique);
  check(sql, {.rows = 380, .known_domain = false}, {"build.parquet"});
}

TEST_CASE_METHOD(DomainEvidenceFixture,
                 "native domain evidence retains the unfiltered catalog bound",
                 "[integration][gpu_execution][dynamic_filter][domain_evidence][build_key_domain]")
{
  {
    scoped_sirius_setting cpu(*con, "gpu_execution", false);
    run_ok("CREATE TABLE domain_native AS SELECT i::BIGINT AS k FROM range(100) t(i);");
  }
  auto const domain =
    evidence("SELECT p.p FROM read_parquet(" + directory.file_literal("probe.parquet") +
             ") p JOIN domain_native b ON p.k = b.k WHERE b.k < 60");
  REQUIRE(domain.cardinality == 100);
  REQUIRE(domain.source == sirius::op::dynamic_filter_domain_source::catalog_bound);
  REQUIRE_FALSE(domain.declared_unique);
}

TEST_CASE_METHOD(DomainEvidenceFixture,
                 "pinned domain evidence requires the complete file set",
                 "[integration][gpu_execution][dynamic_filter][domain_evidence]")
{
  auto const tier      = GENERATE(std::string{"gpu"}, std::string{"host"});
  auto const threshold = GENERATE(std::string{"0.9"}, std::string{"0.45"});
  CAPTURE(tier, threshold);
  scoped_sirius_setting coverage(
    *con, "dynamic_filter_domain_coverage_threshold", std::string_view{threshold});
  build("part-a.parquet", build_keys::distinct, 0, 50);
  build("part-b.parquet", build_keys::distinct, 50, 100);
  pin("part-*.parquet", tier);
  auto const files = "[" + directory.file_literal("part-b.parquet") + ", " +
                     directory.file_literal("part-a.parquet") + "]";
  check(query(files, 95), {.rows = 190, .gated = true}, {"part-a.parquet", "part-b.parquet"});
  check(query(directory.file_literal("part-b.parquet"), 95),
        {.rows = 90, .known_domain = false, .served_from_pin = false},
        {"part-b.parquet"});
  auto const repeated = "[" + directory.file_literal("part-a.parquet") + ", " +
                        directory.file_literal("part-a.parquet") + ", " +
                        directory.file_literal("part-b.parquet") + "]";
  check(query(repeated, 95),
        {.rows = 290, .known_domain = false, .served_from_pin = false},
        {"part-a.parquet", "part-b.parquet"});
}

TEST_CASE_METHOD(DomainEvidenceFixture,
                 "pin replacement and unpin discard old domain declarations",
                 "[integration][gpu_execution][dynamic_filter][domain_evidence]")
{
  auto const tier      = GENERATE(std::string{"gpu"}, std::string{"host"});
  auto const threshold = GENERATE(std::string{"0.9"}, std::string{"0.45"});
  CAPTURE(tier, threshold);
  scoped_sirius_setting coverage(
    *con, "dynamic_filter_domain_coverage_threshold", std::string_view{threshold});
  build("first.parquet");
  build("replacement.parquet");
  pin("first.parquet", tier);
  auto const first       = query(directory.file_literal("first.parquet"), 95);
  auto const replacement = query(directory.file_literal("replacement.parquet"), 95);
  check(first, {.rows = 190, .gated = true}, {"first.parquet"});

  pin("replacement.parquet", tier, declaration::none);
  check(replacement, {.rows = 190}, {"replacement.parquet"});
  check(first, {.rows = 190, .known_domain = false, .served_from_pin = false}, {"first.parquet"});

  run_ok("CALL unpin_table(" + sql_literal(pin_name()) + ");");
  pin("replacement.parquet", tier);
  check(replacement, {.rows = 190, .gated = true}, {"replacement.parquet"});
  run_ok("CALL unpin_table(" + sql_literal(pin_name()) + ");");
  check(replacement,
        {.rows = 190, .known_domain = false, .served_from_pin = false},
        {"replacement.parquet"});
  pin("replacement.parquet", tier, declaration::none);
  check(replacement, {.rows = 190}, {"replacement.parquet"});
}

TEST_CASE_METHOD(DomainEvidenceFixture,
                 "pin declarations reject invalid input before publication",
                 "[integration][gpu_execution][dynamic_filter][domain_evidence][pin_table]")
{
  build("build.parquet");
  auto const args = directory.file_literal("build.parquet") + ", name=" + sql_literal(pin_name());
  for (auto const& sql : std::vector<std::string>{
         "CALL pin_table(" + args + ", tier='gpu', unique_cols=['k', NULL]);",
         "CALL pin_table(" + args + ", tier='gpu', unique_cols='k');",
         "CALL pin_table(" + args + ", tier='parquet', unique_cols=['k']);",
         "CALL pin_table(format='duckdb', name='domain_native', tier='gpu', unique_cols=['k']);"}) {
    INFO(sql);
    auto prepared = con->Prepare(sql);
    REQUIRE(prepared);
    REQUIRE(prepared->HasError());
  }

  auto& manager     = sirius::test::get_registered_sirius_context(*con)->get_scan_manager();
  auto const epoch  = manager.pin_registry_epoch();
  auto const before = sirius::test::get_transparent_execution_stats(*con);
  // Only the source schema and the pinned columns can show these declarations are invalid.
  for (auto const& sql : std::vector<std::string>{
         "CALL pin_table(" + args + ", tier='gpu', unique_cols=['missing']);",
         "CALL pin_table(" + args + ", tier='gpu', cols=['payload'], unique_cols=['k']);"}) {
    INFO(sql);
    auto invalid = con->Query(sql);
    REQUIRE(invalid);
    REQUIRE(invalid->HasError());
    REQUIRE(invalid->GetError().find("unique_cols") != std::string::npos);
  }
  REQUIRE(manager.pin_registry_epoch() == epoch);
  REQUIRE(sirius::test::get_transparent_execution_stats(*con).parquet_reader_calls ==
          before.parquet_reader_calls);
}

TEST_CASE_METHOD(DomainEvidenceFixture,
                 "each declared pin column is an independent usefulness hint",
                 "[integration][gpu_execution][dynamic_filter][domain_evidence][pin_table]")
{
  auto const tier      = GENERATE(std::string{"gpu"}, std::string{"host"});
  auto const threshold = GENERATE(std::string{"0.9"}, std::string{"0.45"});
  CAPTURE(tier, threshold);
  scoped_sirius_setting coverage(
    *con, "dynamic_filter_domain_coverage_threshold", std::string_view{threshold});
  build("build.parquet");
  run_ok("CALL pin_table(" + directory.file_literal("build.parquet") +
         ", tier=" + sql_literal(tier) + ", name=" + sql_literal(pin_name()) +
         ", cols=['marker', 'other', 'payload', 'k'], unique_cols=['K', 'other', 'k']);");
  bool found = false;
  sirius::test::get_registered_sirius_context(*con)->get_scan_manager().visit_pinned_entries(
    [&](std::string_view name, auto const& entry) {
      if (name == pin_name()) {
        found = true;
        CHECK(entry.declared_unique_columns == std::vector<std::string>{"k", "other"});
        CHECK(std::none_of(entry.proven_unique_columns.begin(),
                           entry.proven_unique_columns.end(),
                           [](bool proven) { return proven; }));
      }
      return true;
    });
  REQUIRE(found);
  for (auto const* key : {"k", "other"}) {
    CAPTURE(key);
    check(query(directory.file_literal("build.parquet"), 95, key),
          {.rows = 190, .gated = true},
          {"build.parquet"});
  }
}
