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

// Scans a distributed plan cannot split are refused while lowering, before the plan is
// distributed: no exchange is connected, so none of this needs NIXL or a second process.

#include <catch.hpp>
#include <duckdb.hpp>
#include <duckdb/common/multi_file/multi_file_reader.hpp>
#include <duckdb/common/string_util.hpp>
#include <duckdb/execution/column_binding_resolver.hpp>
#include <duckdb/planner/logical_operator.hpp>
#include <helper/type_conversions.hpp>
#include <op/scan/parquet_scan_types.hpp>
#include <planner/distributed_plan.hpp>
#include <planner/sirius_physical_plan_generator.hpp>
#include <planner/substrait_scan_ranges.hpp>
#include <sirius/exception.hpp>
#include <utils/parquet_fixture_utils.hpp>
#include <utils/sirius_test_env.hpp>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

namespace {

void write_parquet(fs::path const& path)
{
  sirius::test::scoped_sirius_disable disable_sirius;
  duckdb::DuckDB db(nullptr);
  duckdb::Connection con(db);
  auto r = con.Query("COPY (SELECT range AS k FROM range(1000)) TO " +
                     sirius::test::sql_literal(path.string()) + " (FORMAT PARQUET);");
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
       "      usage_limit_fraction: 0.2\n"
       "      reservation_limit_fraction: 1.0\n"
       "    host:\n"
       "      capacity_bytes: 4000000000\n"
       "      initial_number_pools: 2\n"
       "      pool_size: 512\n"
       "      block_size: 1048576\n"
       "  executor:\n"
       "    pipeline:\n"
       "      num_threads: 2\n"
       "    task_creator:\n"
       "      num_threads: 1\n"
       "    downgrade:\n"
       "      num_threads: 1\n";
}

void pause_shared_envs()
{
  for (auto* env : {sirius::test::g_shared_env,
                    sirius::test::g_integration_env,
                    sirius::test::g_integration_env_2gpu}) {
    if (env && env->is_active()) { env->pause(); }
  }
}

class physical_planner : public sirius::planner::sirius_physical_plan_generator {
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

/// Plans @p sql as rank 1 of 2 and lowers its scans; returns what lowering threw, or "".
std::string lowering_error(duckdb::Connection& con, std::string const& sql)
{
  REQUIRE_FALSE(con.Query("SET gpu_execution = false")->HasError());
  // Rank first: with the peers already set, SET sirius_exchange_rank would connect.
  REQUIRE_FALSE(con.Query("SET sirius_exchange_rank = 1")->HasError());
  REQUIRE_FALSE(con.Query("SET sirius_exchange_peers = '127.0.0.1:1,127.0.0.1:2'")->HasError());
  REQUIRE_FALSE(con.Query("BEGIN TRANSACTION READ ONLY")->HasError());
  physical_planner planner(*con.context);
  auto plan = planner.plan(con.ExtractPlan(sql));
  std::string error;
  try {
    planner.insert_gpu_pipeline_operators(plan);
  } catch (sirius::not_implemented_exception const& e) {
    error = e.what();
  }
  REQUIRE_FALSE(con.Query("ROLLBACK")->HasError());
  return error;
}

}  // namespace

TEST_CASE("a distributed plan refuses scans it cannot split", "[planner][distributed]")
{
  pause_shared_envs();
  sirius::test::scratch_dir scratch{"distributed_scan_refusals"};
  auto const parquet = scratch.path() / "t.parquet";
  write_parquet(parquet);
  auto const config = scratch.path() / "sirius.yaml";
  write_config(config);
  sirius::test::shared_test_env env(config);
  auto con = env.make_connection();

  SECTION("a DuckDB table")
  {
    // A file-backed database: Sirius scans DuckDB storage from one.
    REQUIRE_FALSE(
      con
        .Query("ATTACH " + sirius::test::sql_literal((scratch.path() / "t.db").string()) + " AS f")
        ->HasError());
    REQUIRE_FALSE(con.Query("CREATE TABLE f.t AS SELECT range AS a FROM range(10)")->HasError());
    CHECK_THAT(lowering_error(con, "SELECT sum(a) FROM f.t"),
               Catch::Matchers::ContainsSubstring("distributed execution: a DuckDB table"));
  }

  SECTION("sirius_read_parquet of one URI")
  {
    CHECK_THAT(lowering_error(con,
                              "SELECT sum(k) FROM sirius_read_parquet(" +
                                sirius::test::sql_literal(parquet.string()) + ")"),
               Catch::Matchers::ContainsSubstring("read_parquet of one remote URI"));
  }

  SECTION("file_index or filename of a split table")
  {
    auto const source = "read_parquet(" + sirius::test::sql_literal(parquet.string()) + ")";
    auto const sql    = GENERATE(as<std::string>{},
                              "SELECT file_index, count(*) FROM {} GROUP BY 1",
                              "SELECT count(*) FROM {} WHERE filename <> ''");
    CHECK_THAT(
      lowering_error(con, duckdb::StringUtil::Replace(sql, "{}", source)),
      Catch::Matchers::ContainsSubstring("filename or file_index of a table split across ranks"));
  }

  SECTION("a plan that carries its own byte ranges")
  {
    std::map<std::string, std::vector<sirius::planner::scan_byte_range>> ranges;
    ranges[parquet.string()] = {{0, 100}};
    con.context->registered_state->Insert(
      sirius::planner::scan_byte_ranges_state::kStateKey,
      duckdb::make_shared_ptr<sirius::planner::scan_byte_ranges_state>(std::move(ranges)));
    CHECK_THAT(lowering_error(con,
                              "SELECT sum(k) FROM read_parquet(" +
                                sirius::test::sql_literal(parquet.string()) + ")"),
               Catch::Matchers::ContainsSubstring("a plan that carries its own byte ranges"));
    con.context->registered_state->Remove(sirius::planner::scan_byte_ranges_state::kStateKey);
  }
}

TEST_CASE("assign_files_to_rank takes this rank's share of a table's files",
          "[planner][distributed]")
{
  using sirius::planner::assign_files_to_rank;
  sirius::operator_params params;
  auto const as_rank = [&](std::uint32_t rank, std::size_t world) {
    params.exchange_peers = std::vector<std::string>(world, "127.0.0.1:1");
    params.exchange_rank  = rank;
  };

  SECTION("one rank reads every file whole")
  {
    sirius::op::scan::parquet_ingestible_table_info info;
    assign_files_to_rank(info, {"/a", "/b"}, params);
    CHECK(info.resolved_file_paths == std::vector<std::string>{"/a", "/b"});
    CHECK(info.resolved_file_ranges.empty());
  }

  SECTION("remote files are dealt whole, rank, rank + world, ...")
  {
    as_rank(1, 2);
    sirius::op::scan::parquet_ingestible_table_info info;
    assign_files_to_rank(info, {"s3://b/0", "s3://b/1", "s3://b/2", "s3://b/3"}, params);
    CHECK(info.resolved_file_paths == std::vector<std::string>{"s3://b/1", "s3://b/3"});
    CHECK(info.resolved_file_ranges.empty());
  }

  SECTION("a remote table needs a file per rank")
  {
    as_rank(0, 3);
    sirius::op::scan::parquet_ingestible_table_info info;
    CHECK_THROWS_AS(assign_files_to_rank(info, {"s3://b/0", "s3://b/1"}, params),
                    sirius::not_implemented_exception);
  }

  SECTION("local files are split by byte range, and a rank may get none")
  {
    sirius::test::scratch_dir scratch{"assign_files_to_rank"};
    auto const file = (scratch.path() / "two_bytes").string();
    std::ofstream(file) << "xy";
    // 2 bytes over 3 ranks: rank 0's span is empty, ranks 1 and 2 get a byte each.
    for (std::uint32_t rank = 0; rank < 3; ++rank) {
      as_rank(rank, 3);
      sirius::op::scan::parquet_ingestible_table_info info;
      assign_files_to_rank(info, {file}, params);
      if (rank == 0) {
        CHECK(info.resolved_file_paths.empty());
      } else {
        CHECK(info.resolved_file_paths == std::vector<std::string>{file});
        REQUIRE(info.resolved_file_ranges.size() == 1);
        CHECK(info.resolved_file_ranges[0] ==
              std::vector<std::pair<std::uint64_t, std::uint64_t>>{{rank - 1, 1}});
      }
    }
  }

  SECTION("a scan that reads file_index is refused")
  {
    as_rank(0, 2);
    sirius::op::scan::parquet_ingestible_table_info info;
    info.virtual_columns.push_back(sirius::op::scan::bound_virtual_column{
      duckdb::MultiFileReader::COLUMN_IDENTIFIER_FILE_INDEX,
      "file_index",
      sirius::from_duckdb(duckdb::LogicalType::UBIGINT),
      sirius::op::scan::scan_plan::parquet_virtual_column_kind::FILE_INDEX});
    info.column_ids.emplace_back(duckdb::MultiFileReader::COLUMN_IDENTIFIER_FILE_INDEX);
    CHECK_THROWS_AS(assign_files_to_rank(info, {"s3://b/0", "s3://b/1"}, params),
                    sirius::not_implemented_exception);
  }
}
