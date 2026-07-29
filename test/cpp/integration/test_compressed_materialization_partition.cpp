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

// End-to-end coverage of narrow payload carriers crossing an engaged partitioned exchange.
// The fixture forces multi-way hash partitioning on one GPU (tiny hash_partition_bytes) and
// wires a dynamic filter onto the probe scan's join key (filtered build side), so it exercises
// the column-granular dynamic-filter guard and the tier narrowing policy together: the join-key
// column goes native at the scan (on GPU tier the policy retracts it as a boundary restore, and
// it is independently a guard target) while the payload columns — join-payload transport the
// policy keeps — stay narrow through scan -> DYNAMIC_FILTER -> CONCAT -> PARTITION -> hash
// join. The partition_narrow_columns counter — derived from actual batch types inside the
// hash-partition path — is the engagement proof; a plan-shape case pins down the sidecar
// stamps on the DYNAMIC_FILTER / GPU_SCAN leaf pair and the wrap-time copies, and a grouped
// aggregation case proves narrow group keys cross the aggregate-side exchange. An outer-join case
// covers the join types that are forced onto the STANDARD partitioned path, including a side that
// yields no row at run time.

#include "compressed_materialization_test_common.hpp"
#include "cudf/cudf_utils.hpp"
#include "op/scan/sirius_physical_dynamic_filter.hpp"
#include "op/sirius_physical_concat.hpp"
#include "op/sirius_physical_operator.hpp"
#include "op/sirius_physical_partition.hpp"
#include "planner/sirius_physical_plan_generator.hpp"
#include "sirius_context.hpp"

#include <cudf/types.hpp>
#include <cudf/utilities/traits.hpp>

#include <catch.hpp>
#include <duckdb.hpp>
#include <duckdb/execution/column_binding_resolver.hpp>
#include <duckdb/main/config.hpp>
#include <duckdb/optimizer/optimizer.hpp>
#include <duckdb/parser/parser.hpp>
#include <duckdb/planner/planner.hpp>
#include <unistd.h>
#include <utils/sirius_test_env.hpp>
#include <utils/transparent_execution_test_utils.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace fs = std::filesystem;

using sirius::op::sirius_physical_operator;
using sirius::op::SiriusPhysicalOperatorType;

using namespace sirius::test::compmat;

namespace {

// Fact table: 200000 rows of (k BIGINT, v BIGINT, d DECIMAL(18,2)). k = range % 50000 pins
// narrow as INT32 (join key / group key), v = (range * 11) % 297 as INT16, and d's unscaled
// values stay under 29000, picking DECIMAL32. ROW_GROUP_SIZE is a 2048-row multiple (DuckDB
// rounds parquet row groups up to vector-size multiples).
constexpr std::int64_t kFactRows = 200000;
// Dimension table: (k BIGINT = range, x BIGINT = range % 100). The x < 50 filter makes the
// build side filtered, which wires a dynamic-filter producer onto the probe scan's k.
constexpr std::int64_t kDimRows = 50000;

// The filtered build side (~25000 rows of one BIGINT key ~ 200 KB) exceeds
// hash_partition_bytes several times over, so the join's natural partition count is >= 2; the
// aggregate-side partial output (50000 groups) engages the aggregate partition the same way.
// kMaxBuildHashTableBytes sits below the build bytes so the small build cannot elect
// BUILD_PROBE/broadcast, which would stream the probe through its PARTITION unpartitioned and
// bypass the hash-partition path the partition_narrow_columns counter observes.
constexpr std::size_t kScanBatchBytes         = 262144;
constexpr std::size_t kHashPartitionBytes     = 65536;
constexpr std::size_t kMaxBuildHashTableBytes = 1024;

constexpr char const* kJoinQuery =
  "SELECT SUM(t.v) AS sv, SUM(t.d) AS sd FROM t JOIN dm ON t.k = dm.k WHERE dm.x < 50;";

// RIGHT and FULL OUTER joins are forced onto the STANDARD partitioned path, so they exercise the
// per-side input schema the join reads whenever it has to build a side's batch itself.
//
// The gap table supplies a side that yields no row at run time without being provably empty at plan
// time: a statically empty side is folded out of the plan by DuckDB, taking the join with it. Its
// keys are the 4096 values [0, 4095] plus the 4096 values [104096, 108191], written 2048 rows per
// row group, so no row group's range contains kGapUnmatchedKey while the file-level range does.
constexpr std::int64_t kGapRows         = 8192;
constexpr std::int64_t kGapUnmatchedKey = 50000;
constexpr std::size_t kGapRowGroupSize  = 2048;

void generate_gap_parquet(fs::path const& path)
{
  generate_parquet(path,
                   "SELECT CASE WHEN range < 4096 THEN range ELSE range + 100000 END AS k, "
                   "(range * 11) % 297 AS v FROM range(" +
                     std::to_string(kGapRows) + ")",
                   kGapRowGroupSize);
}

// An outer join over two fully pinned narrow tables. Both payload carriers survive to the join as
// join-payload transport, so each of the join's input ports carries a narrow schema.
constexpr char const* kOuterJoinQuery =
  "SELECT SUM(t.v) AS sv, SUM(t.d) AS sd, COUNT(dm.x) AS cx FROM t FULL OUTER JOIN dm ON t.k = "
  "dm.k;";

/// Outer joins whose g side yields no row. Aggregated so the result comparison stays cheap while
/// the join still materializes every surviving row; the counts cover the NULL padding the join
/// emits for the side that contributed nothing.
std::vector<std::string> runtime_empty_side_queries()
{
  std::string const dead = "(SELECT * FROM g WHERE k = " + std::to_string(kGapUnmatchedKey) + ")";
  return {"SELECT COUNT(*) AS c, COUNT(a.v) AS cav, COUNT(b.x) AS cbx FROM " + dead +
            " a FULL OUTER JOIN dm b ON a.k = b.k;",
          "SELECT COUNT(*) AS c, COUNT(a.v) AS cav, COUNT(b.x) AS cbx FROM " + dead +
            " a RIGHT JOIN dm b ON a.k = b.k;",
          "SELECT COUNT(*) AS c, COUNT(a.v) AS cav, COUNT(t.v) AS ctv FROM t LEFT JOIN " + dead +
            " a ON t.k = a.k;"};
}

constexpr char const* kGroupByQuery =
  "SELECT k, COUNT(*) AS c, SUM(v) AS sv FROM t GROUP BY k ORDER BY k LIMIT 10;";
constexpr char const* kCountValidGroupByQuery =
  "SELECT k, COUNT(k) AS c FROM t GROUP BY k ORDER BY k LIMIT 10;";

// Small exchange sizes force multi-way hash partitioning on the fixture's data volumes.
constexpr config_values kConfigValues{.scan_batch_bytes           = kScanBatchBytes,
                                      .hash_partition_bytes       = kHashPartitionBytes,
                                      .concat_batch_bytes         = 262144,
                                      .max_build_hash_table_bytes = kMaxBuildHashTableBytes};

void generate_fact_parquet(fs::path const& path)
{
  generate_parquet(path,
                   "SELECT range % 50000 AS k, (range * 11) % 297 AS v, "
                   "CAST(((range * 13) % 29000) / 100.0 AS DECIMAL(18,2)) AS d "
                   "FROM range(" +
                     std::to_string(kFactRows) + ")",
                   30720);
}

void generate_dim_parquet(fs::path const& path)
{
  generate_parquet(
    path,
    "SELECT range AS k, range % 100 AS x FROM range(" + std::to_string(kDimRows) + ")",
    10240);
}

/// Build the Sirius physical plan for @p query exactly as the transparent path does
/// (parse/plan/optimize with the production-disabled optimizers, then
/// sirius_physical_plan_generator::create_plan). Restores the optimizer settings on exit.
duckdb::unique_ptr<sirius_physical_operator> build_sirius_plan(duckdb::Connection& con,
                                                               std::string const& query)
{
  auto& context = *con.context;

  auto original_disabled = duckdb::DBConfig::GetConfig(context).options.disabled_optimizers;
  auto& disabled         = duckdb::DBConfig::GetConfig(context).options.disabled_optimizers;
  disabled.insert(duckdb::OptimizerType::IN_CLAUSE);
  disabled.insert(duckdb::OptimizerType::COMPRESSED_MATERIALIZATION);
  disabled.insert(duckdb::OptimizerType::LATE_MATERIALIZATION);

  con.Query("BEGIN TRANSACTION");

  duckdb::unique_ptr<sirius_physical_operator> result;
  try {
    duckdb::Parser parser(context.GetParserOptions());
    parser.ParseQuery(query);
    REQUIRE(!parser.statements.empty());

    duckdb::Planner planner(context);
    planner.CreatePlan(std::move(parser.statements[0]));
    REQUIRE(planner.plan);

    auto plan = std::move(planner.plan);
    if (context.config.enable_optimizer) {
      duckdb::Optimizer optimizer(*planner.binder, context);
      plan = optimizer.Optimize(std::move(plan));
    }
    plan->ResolveOperatorTypes();

    duckdb::ColumnBindingResolver resolver;
    resolver.VisitOperator(*plan);

    sirius::planner::sirius_physical_plan_generator gen(context);
    result = gen.create_plan(std::move(plan));
  } catch (...) {
    con.Query("ROLLBACK");
    duckdb::DBConfig::GetConfig(context).options.disabled_optimizers = original_disabled;
    throw;
  }

  con.Query("COMMIT");
  duckdb::DBConfig::GetConfig(context).options.disabled_optimizers = original_disabled;
  return result;
}

template <typename Fn>
void for_each_operator(sirius_physical_operator* root, Fn const& fn)
{
  if (!root) { return; }
  fn(root);
  for (auto& child : root->children) {
    for_each_operator(child.get(), fn);
  }
}

std::vector<sirius_physical_operator*> collect(sirius_physical_operator* root,
                                               SiriusPhysicalOperatorType type)
{
  std::vector<sirius_physical_operator*> out;
  for_each_operator(root, [&](sirius_physical_operator* op) {
    if (op->type == type) { out.push_back(op); }
  });
  return out;
}

bool subtree_contains(sirius_physical_operator* root, sirius_physical_operator const* target)
{
  bool found = false;
  for_each_operator(root, [&](sirius_physical_operator* op) {
    if (op == target) { found = true; }
  });
  return found;
}

std::size_t count_type(std::vector<cudf::data_type> const& sidecar, cudf::type_id id)
{
  return static_cast<std::size_t>(std::count_if(
    sidecar.begin(), sidecar.end(), [id](cudf::data_type t) { return t.id() == id; }));
}

}  // namespace

// NB: no [integration]/[shared_context] tag — these TEST_CASEs build their own SiriusContext
// from a small-partition yaml and manage (pause) the shared envs themselves, mirroring
// test_compressed_materialization_gate.cpp.
TEST_CASE(
  "gpu_execution - narrow payloads cross a partitioned exchange with a wired dynamic "
  "filter",
  "[gpu_execution][parquet][compressed_materialization_partition]")
{
  pause_shared_envs();

  auto tmp = fs::temp_directory_path() / ("sirius-compmat-part-" + std::to_string(::getpid()));
  std::error_code ec;
  fs::remove_all(tmp, ec);
  fs::create_directories(tmp);

  auto fact_path = tmp / "fact.parquet";
  generate_fact_parquet(fact_path);
  auto dim_path = tmp / "dim.parquet";
  generate_dim_parquet(dim_path);

  auto yaml_path = tmp / "compmat_partition.yaml";
  write_config(yaml_path, kConfigValues);
  REQUIRE(fs::exists(yaml_path));

  {
    sirius::test::shared_test_env local_env(yaml_path);
    auto con = local_env.make_connection();

    require_ok(con.Query("SET enable_duckdb_fallback = false;"), "disable fallback");
    require_ok(
      con.Query("CREATE VIEW t AS SELECT * FROM read_parquet('" + fact_path.string() + "');"),
      "create fact view");
    require_ok(
      con.Query("CREATE VIEW dm AS SELECT * FROM read_parquet('" + dim_path.string() + "');"),
      "create dim view");

    for (auto const* tier : {"gpu", "host"}) {
      DYNAMIC_SECTION("tier = " << tier)
      {
        require_ok(con.Query("SET enable_compressed_materialization = true;"), "enable flag");
        auto const pin_before = sirius::test::get_compressed_materialization_stats(con);
        require_ok(con.Query("CALL pin_table('" + fact_path.string() + "', tier='" +
                             std::string(tier) + "', name='t');"),
                   "pin fact");
        require_ok(con.Query("CALL pin_table('" + dim_path.string() + "', tier='" +
                             std::string(tier) + "', name='dm');"),
                   "pin dim");
        auto const pin_after = sirius::test::get_compressed_materialization_stats(con);
        REQUIRE(pin_after.pin_columns_narrowed > pin_before.pin_columns_narrowed);

        // Flag-on: the join keys are native at the scans (retracted by the tier policy on GPU
        // tier, and t.k independently forced by the guard) while the payloads v and d — join
        // transport the policy keeps on both tiers — stay narrow and cross the engaged
        // probe-side hash partition. The pinned-narrow serve is cast-free (narrowed counter
        // flat) while the keys restore to native during scan normalization.
        bool const gpu_tier = std::string_view(tier) == "gpu";
        auto const before   = sirius::test::get_compressed_materialization_stats(con);
        compare_gpu_vs_cpu(con, kJoinQuery);
        auto const after = sirius::test::get_compressed_materialization_stats(con);
        REQUIRE(after.scan_sidecars_installed > before.scan_sidecars_installed);
        REQUIRE(after.partition_narrow_columns > before.partition_narrow_columns);
        REQUIRE(after.scan_columns_narrowed == before.scan_columns_narrowed);
        REQUIRE(after.scan_columns_restored > before.scan_columns_restored);
        if (gpu_tier) {
          REQUIRE(after.scan_narrow_targets_retracted > before.scan_narrow_targets_retracted);
        } else {
          REQUIRE(after.scan_narrow_targets_retracted == before.scan_narrow_targets_retracted);
        }

        // Flag-off contrast: no sidecars exist, so the narrow cache restores fully at the scan
        // and nothing narrow reaches the exchange.
        require_ok(con.Query("SET enable_compressed_materialization = false;"), "disable flag");
        auto const off_before = sirius::test::get_compressed_materialization_stats(con);
        compare_gpu_vs_cpu(con, kJoinQuery);
        auto const off_after = sirius::test::get_compressed_materialization_stats(con);
        REQUIRE(off_after.partition_narrow_columns == off_before.partition_narrow_columns);
        REQUIRE(off_after.scan_columns_restored > off_before.scan_columns_restored);
        require_ok(con.Query("SET enable_compressed_materialization = true;"), "restore flag");

        require_ok(con.Query("CALL unpin_table('dm');"), "unpin dim");
        require_ok(con.Query("CALL unpin_table('t');"), "unpin fact");
      }
    }
  }

  fs::remove_all(tmp, ec);
}

TEST_CASE("gpu_execution - outer joins over narrow carriers match the CPU",
          "[gpu_execution][parquet][compressed_materialization_partition]")
{
  pause_shared_envs();

  auto tmp = fs::temp_directory_path() / ("sirius-compmat-outer-" + std::to_string(::getpid()));
  std::error_code ec;
  fs::remove_all(tmp, ec);
  fs::create_directories(tmp);

  auto fact_path = tmp / "fact.parquet";
  generate_fact_parquet(fact_path);
  auto dim_path = tmp / "dim.parquet";
  generate_dim_parquet(dim_path);
  auto gap_path = tmp / "gap.parquet";
  generate_gap_parquet(gap_path);

  auto yaml_path = tmp / "compmat_outer.yaml";
  write_config(yaml_path, kConfigValues);
  REQUIRE(fs::exists(yaml_path));

  {
    sirius::test::shared_test_env local_env(yaml_path);
    auto con = local_env.make_connection();

    require_ok(con.Query("SET enable_duckdb_fallback = false;"), "disable fallback");
    require_ok(
      con.Query("CREATE VIEW t AS SELECT * FROM read_parquet('" + fact_path.string() + "');"),
      "create fact view");
    require_ok(
      con.Query("CREATE VIEW dm AS SELECT * FROM read_parquet('" + dim_path.string() + "');"),
      "create dim view");
    require_ok(
      con.Query("CREATE VIEW g AS SELECT * FROM read_parquet('" + gap_path.string() + "');"),
      "create gap view");

    // HOST tier keeps the tier narrowing policy inert, so the narrow carriers the pins recorded
    // survive to the join.
    require_ok(con.Query("SET enable_compressed_materialization = true;"), "enable flag");
    require_ok(con.Query("CALL pin_table('" + fact_path.string() + "', tier='host', name='t');"),
               "pin fact");
    require_ok(con.Query("CALL pin_table('" + dim_path.string() + "', tier='host', name='dm');"),
               "pin dim");
    require_ok(con.Query("CALL pin_table('" + gap_path.string() + "', tier='host', name='g');"),
               "pin gap");

    SECTION("the join's per-side input schema is the narrow one")
    {
      // What the join reads when it has to build one side's batch itself: that child's own
      // physical sidecar, copied onto the CONCAT/PARTITION feeder chain at wrap time. The schema on
      // that port is narrower than the native mapping of the same logical columns, so deriving an
      // empty batch from the logical types instead would not reproduce it.
      auto plan = build_sirius_plan(con, kOuterJoinQuery);
      REQUIRE(plan);
      auto joins = collect(plan.get(), SiriusPhysicalOperatorType::HASH_JOIN);
      REQUIRE(joins.size() == 1);
      REQUIRE(joins[0]->children.size() == 2);

      std::size_t narrow_inputs = 0;
      for (auto const& child : joins[0]->children) {
        REQUIRE(child->has_physical_overrides());
        auto const& physical = child->get_physical_types();
        REQUIRE(physical.size() == child->get_types().size());
        for (std::size_t column_idx = 0; column_idx < physical.size(); column_idx++) {
          auto const native = sirius::get_cudf_type(child->get_types()[column_idx]);
          if (cudf::size_of(physical[column_idx]) < cudf::size_of(native)) { narrow_inputs++; }
        }
      }
      REQUIRE(narrow_inputs > 0);
    }

    SECTION("results match the CPU when one side yields no row")
    {
      for (auto const& query : runtime_empty_side_queries()) {
        INFO("query: " << query);
        auto const before = sirius::test::get_compressed_materialization_stats(con);
        compare_gpu_vs_cpu(con, query);
        auto const after = sirius::test::get_compressed_materialization_stats(con);
        // The residency gate installed a narrow sidecar for this query, so the comparison above is
        // a flag-on result over narrow carriers rather than a silently inert one.
        REQUIRE(after.scan_sidecars_installed > before.scan_sidecars_installed);
        REQUIRE(after.scan_narrow_targets_retracted == before.scan_narrow_targets_retracted);
      }
    }

    require_ok(con.Query("CALL unpin_table('g');"), "unpin gap");
    require_ok(con.Query("CALL unpin_table('dm');"), "unpin dim");
    require_ok(con.Query("CALL unpin_table('t');"), "unpin fact");
  }

  fs::remove_all(tmp, ec);
}

TEST_CASE("plan shape - dynamic-filter-wired scan keeps narrow payload carriers",
          "[compressed_materialization_partition][planner]")
{
  pause_shared_envs();

  auto tmp = fs::temp_directory_path() / ("sirius-compmat-shape-" + std::to_string(::getpid()));
  std::error_code ec;
  fs::remove_all(tmp, ec);
  fs::create_directories(tmp);

  auto fact_path = tmp / "fact.parquet";
  generate_fact_parquet(fact_path);
  auto dim_path = tmp / "dim.parquet";
  generate_dim_parquet(dim_path);

  auto yaml_path = tmp / "compmat_shape.yaml";
  write_config(yaml_path, kConfigValues);
  REQUIRE(fs::exists(yaml_path));

  {
    sirius::test::shared_test_env local_env(yaml_path);
    auto con = local_env.make_connection();

    require_ok(con.Query("SET enable_duckdb_fallback = false;"), "disable fallback");
    require_ok(con.Query("SET enable_compressed_materialization = true;"), "enable flag");
    require_ok(
      con.Query("CREATE VIEW t AS SELECT * FROM read_parquet('" + fact_path.string() + "');"),
      "create fact view");
    require_ok(
      con.Query("CREATE VIEW dm AS SELECT * FROM read_parquet('" + dim_path.string() + "');"),
      "create dim view");
    require_ok(con.Query("CALL pin_table('" + fact_path.string() + "', tier='gpu', name='t');"),
               "pin fact");
    require_ok(con.Query("CALL pin_table('" + dim_path.string() + "', tier='gpu', name='dm');"),
               "pin dim");

    auto plan = build_sirius_plan(con, kJoinQuery);
    REQUIRE(plan);

    // The filtered build side wired a dynamic filter onto t.k, so exactly one DYNAMIC_FILTER
    // wraps the fact scan. Its sidecar keeps the payloads narrow (one INT16 v, one DECIMAL32 d)
    // with every remaining entry native — including the planned target key k (INT64) — and its
    // GPU_SCAN child carries the same sidecar.
    auto dynamic_filters = collect(plan.get(), SiriusPhysicalOperatorType::DYNAMIC_FILTER);
    REQUIRE(dynamic_filters.size() == 1);
    auto* dynamic_filter = dynamic_filters.front();
    auto const sidecar   = dynamic_filter->get_physical_types();
    REQUIRE(!sidecar.empty());
    REQUIRE(count_type(sidecar, cudf::type_id::INT16) == 1);
    REQUIRE(count_type(sidecar, cudf::type_id::DECIMAL32) == 1);
    REQUIRE(count_type(sidecar, cudf::type_id::INT64) == sidecar.size() - 2);
    REQUIRE(dynamic_filter->children.size() == 1);
    auto* gpu_scan = dynamic_filter->children[0].get();
    REQUIRE(gpu_scan->type == SiriusPhysicalOperatorType::GPU_SCAN);
    REQUIRE(gpu_scan->get_physical_types() == sidecar);

    // The probe-side CONCAT -> PARTITION wrappers above the DYNAMIC_FILTER carry the sidecar of
    // the subtree they wrap (an optional pure-reference projection between PARTITION and
    // DYNAMIC_FILTER is tolerated — its sidecar forwards the same carriers).
    sirius_physical_operator* probe_concat = nullptr;
    for (auto* concat : collect(plan.get(), SiriusPhysicalOperatorType::CONCAT)) {
      if (subtree_contains(concat, dynamic_filter)) { probe_concat = concat; }
    }
    REQUIRE(probe_concat != nullptr);
    REQUIRE(probe_concat->children.size() == 1);
    auto* probe_partition = probe_concat->children[0].get();
    REQUIRE(probe_partition->type == SiriusPhysicalOperatorType::PARTITION);
    auto const wrapped_sidecar = probe_partition->children[0]->get_physical_types();
    REQUIRE(probe_concat->get_physical_types() == wrapped_sidecar);
    REQUIRE(probe_partition->get_physical_types() == wrapped_sidecar);
    REQUIRE(count_type(wrapped_sidecar, cudf::type_id::INT16) == 1);
    REQUIRE(count_type(wrapped_sidecar, cudf::type_id::DECIMAL32) == 1);
    REQUIRE(count_type(wrapped_sidecar, cudf::type_id::INT64) == wrapped_sidecar.size() - 2);

    require_ok(con.Query("CALL unpin_table('dm');"), "unpin dim");
    require_ok(con.Query("CALL unpin_table('t');"), "unpin fact");
  }

  fs::remove_all(tmp, ec);
}

TEST_CASE("gpu_execution - narrow group keys cross the aggregate exchange",
          "[gpu_execution][parquet][compressed_materialization_partition]")
{
  pause_shared_envs();

  auto tmp = fs::temp_directory_path() / ("sirius-compmat-groupby-" + std::to_string(::getpid()));
  std::error_code ec;
  fs::remove_all(tmp, ec);
  fs::create_directories(tmp);

  auto fact_path = tmp / "fact.parquet";
  generate_fact_parquet(fact_path);

  auto yaml_path = tmp / "compmat_groupby.yaml";
  write_config(yaml_path, kConfigValues);
  REQUIRE(fs::exists(yaml_path));

  {
    sirius::test::shared_test_env local_env(yaml_path);
    auto con = local_env.make_connection();

    require_ok(con.Query("SET enable_duckdb_fallback = false;"), "disable fallback");
    require_ok(
      con.Query("CREATE VIEW t AS SELECT * FROM read_parquet('" + fact_path.string() + "');"),
      "create fact view");
    require_ok(con.Query("SET enable_compressed_materialization = true;"), "enable flag");
    require_ok(con.Query("CALL pin_table('" + fact_path.string() + "', tier='gpu', name='t');"),
               "pin fact");

    // Flag-on: the narrow INT32 group key k earns group-key transport (the tier policy keeps it
    // on this GPU-tier pin) and crosses the engaged aggregate-side partition (50000 groups of
    // partial output against the small hash_partition_bytes). This fails if the HASH_GROUP_BY
    // propagation case falls back to the native boundary or the aggregate wrap drops the
    // sidecar. v is an aggregate input — a boundary restore the policy retracts — so it emits
    // native at the scan.
    auto const before = sirius::test::get_compressed_materialization_stats(con);
    compare_gpu_vs_cpu(con, kGroupByQuery);
    auto const after = sirius::test::get_compressed_materialization_stats(con);
    REQUIRE(after.scan_sidecars_installed > before.scan_sidecars_installed);
    REQUIRE(after.partition_narrow_columns > before.partition_narrow_columns);
    REQUIRE(after.scan_narrow_targets_retracted > before.scan_narrow_targets_retracted);

    // COUNT_VALID reads only k's validity mask, so counting the group key must not force its value
    // carrier native before the aggregate-side exchange; with k the only scanned column and kept
    // narrow, the tier policy retracts nothing.
    auto const count_before = sirius::test::get_compressed_materialization_stats(con);
    compare_gpu_vs_cpu(con, kCountValidGroupByQuery);
    auto const count_after = sirius::test::get_compressed_materialization_stats(con);
    REQUIRE(count_after.scan_sidecars_installed > count_before.scan_sidecars_installed);
    REQUIRE(count_after.partition_narrow_columns > count_before.partition_narrow_columns);
    REQUIRE(count_after.scan_narrow_targets_retracted ==
            count_before.scan_narrow_targets_retracted);

    // Flag-off contrast: everything restores at the scan; the exchange sees native batches.
    require_ok(con.Query("SET enable_compressed_materialization = false;"), "disable flag");
    auto const off_before = sirius::test::get_compressed_materialization_stats(con);
    compare_gpu_vs_cpu(con, kGroupByQuery);
    auto const off_after = sirius::test::get_compressed_materialization_stats(con);
    REQUIRE(off_after.partition_narrow_columns == off_before.partition_narrow_columns);

    require_ok(con.Query("CALL unpin_table('t');"), "unpin fact");
  }

  fs::remove_all(tmp, ec);
}
