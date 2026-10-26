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

// Regression test: the plan generator must plan `sirius_read_parquet` — the internal
// rewrite target for read_parquet('s3://…') inside gpu_execution.
// Before the fix, wrap_table_scan_source threw "Unsupported scan function:
// sirius_read_parquet", so S3 SQL failed during planning (and S3 has no CPU fallback;
// the CPU-side table function only throws). The scan is URI-agnostic — the resolved
// URI travels in parameters[0] — so a local parquet file exercises the exact same plan
// path without an S3 harness. A passing query also proves the GPU path served it: the
// DuckDB-side execute callback for this function unconditionally throws.

#include <catch.hpp>
#include <duckdb.hpp>
#include <utils/parquet_fixture_utils.hpp>
#include <utils/sirius_test_env.hpp>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>

namespace fs = std::filesystem;

namespace {

constexpr std::int64_t kRows = 100'000;

// Throwaway, Sirius-disabled DuckDB writes the parquet so the extension callback does
// not build a SiriusContext on it.
void generate_parquet(fs::path const& path)
{
  sirius::test::scoped_sirius_disable disable_sirius;
  duckdb::DuckDB gen_db(nullptr);
  duckdb::Connection gen(gen_db);
  auto r = gen.Query("COPY (SELECT range AS k, range * 2 AS v FROM range(" + std::to_string(kRows) +
                     ")) TO " + sirius::test::sql_literal(path.string()) + " (FORMAT PARQUET);");
  REQUIRE(r);
  REQUIRE_FALSE(r->HasError());
}

void write_config(fs::path const& yaml_path)
{
  std::ofstream f(yaml_path);
  f << "sirius:\n"
       "  topology:\n"
       "    num_gpus: 1\n"
       "  memory:\n"
       "    gpu:\n"
       "      usage_limit_fraction: 0.4\n"
       "      reservation_limit_fraction: 1.0\n"
       "    host:\n"
       "      capacity_bytes: 32000000000\n"
       "      initial_number_pools: 10\n"
       "      pool_size: 512\n"
       "      block_size: 1048576\n"
       "  executor:\n"
       "    pipeline:\n"
       "      num_threads: 4\n"
       "    task_creator:\n"
       "      num_threads: 2\n"
       "    downgrade:\n"
       "      num_threads: 1\n"
       "      monitor_period: 10ms\n"
       "  operator_params:\n"
       "    scan_task_batch_size: 100000000\n"
       "    max_sort_partition_bytes: 0\n"
       "    hash_partition_bytes: 100000000\n"
       "    concat_batch_bytes: 100000000\n"
       "    max_build_hash_table_bytes: 90000000\n";
}

}  // namespace

// NB: no [integration]/[shared_context] tag — this TEST_CASE builds its own SiriusContext
// and manages (pauses) the shared envs itself, mirroring the isolated-context pin tests.
TEST_CASE("tree pipeline build plans sirius_read_parquet scans",
          "[planner][sirius_read_parquet][tree_pipeline]")
{
  if (sirius::test::g_shared_env && sirius::test::g_shared_env->is_active()) {
    sirius::test::g_shared_env->pause();
  }
  if (sirius::test::g_integration_env && sirius::test::g_integration_env->is_active()) {
    sirius::test::g_integration_env->pause();
  }
  if (sirius::test::g_integration_env_2gpu && sirius::test::g_integration_env_2gpu->is_active()) {
    sirius::test::g_integration_env_2gpu->pause();
  }

  sirius::test::scratch_dir scratch{"srp_tree"};
  auto const& tmp = scratch.path();

  auto parquet_path = tmp / "kv.parquet";
  generate_parquet(parquet_path);
  auto collision_path = tmp / "physical_names.parquet";
  {
    sirius::test::scoped_sirius_disable disable;
    duckdb::DuckDB db(nullptr);
    duckdb::Connection gen(db);
    auto result = gen.Query(
      "COPY (SELECT range AS k, CASE WHEN range = 0 THEN NULL ELSE 'physical' END AS filename, "
      "77::BIGINT AS file_index, 88::INTEGER AS file_row_number FROM range(2)) TO " +
      sirius::test::sql_literal(collision_path.string()) + " (FORMAT PARQUET)");
    REQUIRE(result);
    REQUIRE_FALSE(result->HasError());
  }

  auto yaml_path = tmp / "srp_tree.yaml";
  write_config(yaml_path);
  REQUIRE(fs::exists(yaml_path));

  {
    sirius::test::shared_test_env local_env(yaml_path);
    auto con = local_env.make_connection();

    // A planning failure must surface as an error, not a CPU replay (which would throw
    // the function's "internal rewrite target" error anyway).
    auto fb = con.Query("SET enable_duckdb_fallback = false;");
    REQUIRE(fb);
    REQUIRE_FALSE(fb->HasError());

    auto res = con.Query("SELECT max(k), count(*) FROM sirius_read_parquet(" +
                         sirius::test::sql_literal(parquet_path.string()) + ");");
    REQUIRE(res);
    if (res->HasError()) { UNSCOPED_INFO("sirius_read_parquet query error: " << res->GetError()); }
    REQUIRE_FALSE(res->HasError());  // pre-fix: "Unsupported scan function: sirius_read_parquet"
    REQUIRE(res->Collection().GetValue(0, 0).GetValue<int64_t>() == kRows - 1);
    REQUIRE(res->Collection().GetValue(1, 0).GetValue<int64_t>() == kRows);

    auto const scan =
      "sirius_read_parquet(" + sirius::test::sql_literal(parquet_path.string()) + ")";
    auto virtuals = con.Query("SELECT filename, file_index, file_row_number, k FROM " + scan +
                              " WHERE k IN (1, 8193, 99999) ORDER BY k");
    REQUIRE(virtuals);
    INFO((virtuals->HasError() ? virtuals->GetError() : ""));
    REQUIRE_FALSE(virtuals->HasError());
    REQUIRE(virtuals->RowCount() == 3);
    CHECK(virtuals->GetTypes()[0] == duckdb::LogicalType::VARCHAR);
    CHECK(virtuals->GetTypes()[1] == duckdb::LogicalType::UBIGINT);
    CHECK(virtuals->GetTypes()[2] == duckdb::LogicalType::BIGINT);
    for (duckdb::idx_t row = 0; row < virtuals->RowCount(); ++row) {
      CHECK(virtuals->Collection().GetValue(0, row).ToString() == parquet_path.string());
      CHECK(virtuals->Collection().GetValue(1, row).GetValue<uint64_t>() == 0);
      CHECK(virtuals->Collection().GetValue(2, row).GetValue<int64_t>() ==
            virtuals->Collection().GetValue(3, row).GetValue<int64_t>());
    }

    auto filtered = con.Query("SELECT k FROM " + scan + " WHERE file_row_number = 8193");
    REQUIRE(filtered);
    REQUIRE_FALSE(filtered->HasError());
    REQUIRE(filtered->RowCount() == 1);
    CHECK(filtered->Collection().GetValue(0, 0).GetValue<int64_t>() == 8193);

    // Virtual columns are non-null, including when they are only referenced by
    // a predicate. Exercise the residual path rather than assuming the
    // planner's IS_NOT_NULL exemption also handles virtual bindings.
    for (auto const* name : {"filename", "file_index", "file_row_number"}) {
      CAPTURE(name);
      auto non_null = con.Query("SELECT count(*) FROM " + scan + " WHERE " + name + " IS NOT NULL");
      REQUIRE(non_null);
      INFO((non_null->HasError() ? non_null->GetError() : ""));
      REQUIRE_FALSE(non_null->HasError());
      CHECK(non_null->Collection().GetValue(0, 0).GetValue<int64_t>() == kRows);
      auto only_null = con.Query("SELECT k FROM " + scan + " WHERE " + name + " IS NULL");
      REQUIRE(only_null);
      REQUIRE_FALSE(only_null->HasError());
      CHECK(only_null->RowCount() == 0);
    }

    auto empty = con.Query("SELECT filename, file_index, file_row_number FROM " + scan +
                           " WHERE file_index = 1");
    REQUIRE(empty);
    REQUIRE_FALSE(empty->HasError());
    CHECK(empty->RowCount() == 0);

    auto star = con.Query("SELECT * FROM " + scan + " LIMIT 1");
    REQUIRE(star);
    REQUIRE_FALSE(star->HasError());
    CHECK(star->ColumnCount() == 2);

    // Names alone do not establish virtual identity: real columns win binding,
    // retain their physical types, and may contain nulls.
    auto const collision_scan =
      "sirius_read_parquet(" + sirius::test::sql_literal(collision_path.string()) + ")";
    auto physical = con.Query("SELECT filename, file_index, file_row_number FROM " +
                              collision_scan + " ORDER BY k");
    REQUIRE(physical);
    INFO((physical->HasError() ? physical->GetError() : ""));
    REQUIRE_FALSE(physical->HasError());
    REQUIRE(physical->RowCount() == 2);
    CHECK(physical->GetTypes()[1] == duckdb::LogicalType::BIGINT);
    CHECK(physical->GetTypes()[2] == duckdb::LogicalType::INTEGER);
    CHECK(physical->Collection().GetValue(0, 0).IsNull());
    CHECK(physical->Collection().GetValue(0, 1).ToString() == "physical");
    for (duckdb::idx_t row = 0; row < physical->RowCount(); ++row) {
      CHECK(physical->Collection().GetValue(1, row).GetValue<int64_t>() == 77);
      CHECK(physical->Collection().GetValue(2, row).GetValue<int32_t>() == 88);
    }
    auto physical_nulls = con.Query("SELECT k FROM " + collision_scan + " WHERE filename IS NULL");
    REQUIRE(physical_nulls);
    REQUIRE_FALSE(physical_nulls->HasError());
    REQUIRE(physical_nulls->RowCount() == 1);
    CHECK(physical_nulls->Collection().GetValue(0, 0).GetValue<int64_t>() == 0);
  }
}
