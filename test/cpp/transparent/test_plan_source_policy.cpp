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

#include "exec/stream_plan_bindings.hpp"
#include "helper/type_conversions.hpp"
#include "sirius_context.hpp"
#include "sirius_extension.hpp"
#include "transparent/physical_sirius_execution.hpp"
#include "transparent/plan_source_policy.hpp"
#include "transparent/read_view_registry.hpp"
#include "utils/gpu_execution_fixture.hpp"
#include "utils/isolated_checkpoint_test.hpp"
#include "utils/sirius_test_env.hpp"

#include <catch.hpp>
#include <cucascade/data/data_repository.hpp>
#include <duckdb/catalog/catalog.hpp>
#include <duckdb/catalog/catalog_entry/table_function_catalog_entry.hpp>
#include <duckdb/common/multi_file/multi_file_function.hpp>
#include <duckdb/execution/execution_context.hpp>
#include <duckdb/execution/operator/scan/physical_table_scan.hpp>
#include <duckdb/execution/physical_plan_generator.hpp>
#include <duckdb/main/extension/extension_loader.hpp>
#include <duckdb/main/prepared_statement_data.hpp>
#include <duckdb/parallel/interrupt.hpp>
#include <duckdb/parallel/thread_context.hpp>
#include <duckdb/planner/operator/logical_dummy_scan.hpp>
#include <duckdb/planner/operator/logical_get.hpp>

namespace {
using namespace sirius::transparent;

duckdb::unique_ptr<duckdb::LogicalGet> file_scan(std::string name, std::string path)
{
  auto bind       = duckdb::make_uniq<duckdb::MultiFileBindData>();
  bind->file_list = duckdb::make_shared_ptr<duckdb::SimpleMultiFileList>(
    duckdb::vector<duckdb::OpenFileInfo>{duckdb::OpenFileInfo(std::move(path))});
  duckdb::TableFunction function(std::move(name), {}, nullptr);
  return duckdb::make_uniq<duckdb::LogicalGet>(1,
                                               std::move(function),
                                               std::move(bind),
                                               duckdb::vector<duckdb::LogicalType>{},
                                               duckdb::vector<std::string>{});
}

struct throwing_list final : duckdb::SimpleMultiFileList {
  throwing_list() : SimpleMultiFileList({}) {}
  duckdb::vector<duckdb::OpenFileInfo> GetAllFiles() const override
  {
    throw std::runtime_error("inventory extraction failed");
  }
  duckdb::FileExpandResult GetExpandResult() const override
  {
    throw std::runtime_error("inventory extraction failed");
  }
  duckdb::idx_t GetTotalFileCount() const override
  {
    throw std::runtime_error("inventory extraction failed");
  }
  duckdb::OpenFileInfo GetFile(duckdb::idx_t) const override
  {
    throw std::runtime_error("inventory extraction failed");
  }
};

struct external_child_operator final : duckdb::PhysicalOperator {
  external_child_operator(duckdb::PhysicalPlan& plan,
                          duckdb::PhysicalOperator const* child,
                          bool fail)
    : PhysicalOperator(plan, duckdb::PhysicalOperatorType::EXTENSION, {}, 0),
      child(child),
      fail(fail)
  {
  }
  duckdb::PhysicalOperator const* child;
  bool fail;
  duckdb::vector<duckdb::const_reference<duckdb::PhysicalOperator>> GetChildren() const override
  {
    if (fail) throw std::runtime_error("physical walk failed");
    if (child) return {*child};
    return {};
  }
};
}  // namespace

TEST_CASE("Source policy permits complete empty local and native plans",
          "[transparent][policy][shared_context]")
{
  REQUIRE(sirius::test::g_shared_env);
  auto con = sirius::test::g_shared_env->make_connection();
  REQUIRE_FALSE(con.Query("SET gpu_execution=false")->HasError());
  auto local  = file_scan("unregistered_local", "local.parquet");
  auto policy = derive_plan_source_policy(*local, *con.context);
  REQUIRE(policy.discovery_complete);
  REQUIRE(policy.cpu_replay_permitted());
  REQUIRE(policy.scans.at(0).source == byte_source_class::local_file);
  duckdb::LogicalDummyScan empty(0);
  REQUIRE(derive_plan_source_policy(empty, *con.context).cpu_replay_permitted());
  REQUIRE_FALSE(con.Query("CREATE TEMP TABLE r1_policy_native(id INTEGER)")->HasError());
  REQUIRE_FALSE(con.Query("BEGIN")->HasError());
  auto native        = con.ExtractPlan("SELECT * FROM r1_policy_native");
  auto native_policy = derive_plan_source_policy(*native, *con.context);
  REQUIRE(native_policy.discovery_complete);
  REQUIRE(native_policy.cpu_replay_permitted());
  REQUIRE(native_policy.scans.at(0).source == byte_source_class::duckdb_native);
  REQUIRE_FALSE(con.Query("ROLLBACK")->HasError());
}

TEST_CASE("Source policy forbids S3 from unregistered MultiFile sources",
          "[transparent][policy][shared_context]")
{
  REQUIRE(sirius::test::g_shared_env);
  auto con    = sirius::test::g_shared_env->make_connection();
  auto s3     = file_scan("unregistered_s3", "s3://bucket/data.parquet");
  auto policy = derive_plan_source_policy(*s3, *con.context);
  REQUIRE(policy.discovery_complete);
  REQUIRE_FALSE(policy.cpu_replay_permitted());
  REQUIRE(policy.scans.at(0).source == byte_source_class::sirius_owned_s3);
  REQUIRE(policy.reason().find("unregistered_s3") != std::string::npos);
}

TEST_CASE("Source policy preserves S3 veto beside local and unclassified nodes",
          "[transparent][policy][shared_context]")
{
  REQUIRE(sirius::test::g_shared_env);
  auto con = sirius::test::g_shared_env->make_connection();
  duckdb::LogicalDummyScan root(0);
  root.children.push_back(file_scan("local", "local.parquet"));
  auto unknown =
    duckdb::make_uniq<duckdb::LogicalGet>(2,
                                          duckdb::TableFunction("opaque_source", {}, nullptr),
                                          nullptr,
                                          duckdb::vector<duckdb::LogicalType>{},
                                          duckdb::vector<std::string>{});
  auto alone = derive_plan_source_policy(*unknown, *con.context);
  REQUIRE(alone.discovery_complete);
  REQUIRE(alone.cpu_replay_permitted());
  REQUIRE(alone.scans.at(0).source == byte_source_class::unclassified);
  root.children.push_back(std::move(unknown));
  root.children.push_back(file_scan("remote", "S3://bucket/data.parquet"));
  auto policy = derive_plan_source_policy(root, *con.context);
  REQUIRE(policy.discovery_complete);
  REQUIRE_FALSE(policy.cpu_replay_permitted());
  REQUIRE(policy.reason().find("remote") != std::string::npos);
}

TEST_CASE("Source policy treats failed node extraction as incomplete",
          "[transparent][policy][shared_context]")
{
  REQUIRE(sirius::test::g_shared_env);
  auto con   = sirius::test::g_shared_env->make_connection();
  auto node  = file_scan("broken", "local.parquet");
  auto& bind = node->bind_data->Cast<duckdb::MultiFileBindData>();
  SECTION("missing list") { bind.file_list.reset(); }
  SECTION("throwing list") { bind.file_list = duckdb::make_shared_ptr<throwing_list>(); }
  auto policy = derive_plan_source_policy(*node, *con.context);
  REQUIRE_FALSE(policy.discovery_complete);
  REQUIRE_FALSE(policy.cpu_replay_permitted());
  REQUIRE(policy.reason().find("incomplete") != std::string::npos);
}

TEST_CASE("Source policy visits physical GetChildren and forbids a failed walk",
          "[transparent][policy][shared_context]")
{
  REQUIRE(sirius::test::g_shared_env);
  auto con = sirius::test::g_shared_env->make_connection();
  duckdb::PhysicalPlan plan(duckdb::Allocator::Get(*con.context));
  auto logical = file_scan("hidden_remote", "s3://bucket/data.parquet");
  auto& leaf   = plan.Make<duckdb::PhysicalTableScan>(duckdb::vector<duckdb::LogicalType>{},
                                                    logical->function,
                                                    std::move(logical->bind_data),
                                                    duckdb::vector<duckdb::LogicalType>{},
                                                    duckdb::vector<duckdb::ColumnIndex>{},
                                                    duckdb::vector<duckdb::idx_t>{},
                                                    duckdb::vector<std::string>{},
                                                    nullptr,
                                                    0,
                                                    duckdb::ExtraOperatorInfo{},
                                                    duckdb::vector<duckdb::Value>{},
                                                    duckdb::virtual_column_map_t{});
  external_child_operator root(plan, &leaf, false);
  REQUIRE(root.children.empty());
  auto policy = derive_plan_source_policy(root, *con.context);
  REQUIRE(policy.discovery_complete);
  REQUIRE_FALSE(policy.cpu_replay_permitted());
  REQUIRE(policy.reason().find("hidden_remote") != std::string::npos);
  root.fail = true;
  policy    = derive_plan_source_policy(root, *con.context);
  REQUIRE_FALSE(policy.discovery_complete);
  REQUIRE_FALSE(policy.cpu_replay_permitted());
}

TEST_CASE("Source policy forbids verified stream and Sirius S3 sources",
          "[transparent][policy][shared_context]")
{
  REQUIRE(sirius::test::g_shared_env);
  auto con = sirius::test::g_shared_env->make_connection();
  REQUIRE_FALSE(con.Query("SET gpu_execution=false")->HasError());
  REQUIRE_FALSE(con.Query("BEGIN")->HasError());
  for (std::string name : {"sirius_stream_source", "sirius_read_parquet"}) {
    auto& entry =
      duckdb::Catalog::GetSystemCatalog(*con.context)
        .GetEntry<duckdb::TableFunctionCatalogEntry>(*con.context, DEFAULT_SCHEMA, name);
    duckdb::unique_ptr<duckdb::FunctionData> bind;
    if (name == "sirius_stream_source") {
      bind = duckdb::make_uniq<sirius::exec::stream_source_bind_data>(42);
    } else {
      bind = duckdb::make_uniq<duckdb::SiriusReadParquetBindData>("s3://bucket/data.parquet", 1);
    }
    duckdb::LogicalGet node(1, entry.functions.GetFunctionByOffset(0), std::move(bind), {}, {});
    auto policy = derive_plan_source_policy(node, *con.context);
    REQUIRE(policy.discovery_complete);
    REQUIRE_FALSE(policy.cpu_replay_permitted());
    if (name == "sirius_stream_source") {
      REQUIRE(policy.scans.at(0).source == byte_source_class::stream);
      REQUIRE(policy.reason().find("stream source has no CPU body") != std::string::npos);
    } else {
      REQUIRE(policy.scans.at(0).source == byte_source_class::sirius_owned_s3);
    }
  }
  REQUIRE_FALSE(con.Query("ROLLBACK")->HasError());
}

TEST_CASE("Source policy preserves S3 error text and stream reason", "[transparent][policy]")
{
  read_view_comparison mismatch;
  mismatch.correspondence     = "none";
  mismatch.reason             = "no_correspondence";
  auto const mismatch_message = describe_read_view_mismatch(mismatch);

  plan_source_policy policy;
  policy.scans.push_back({"remote", byte_source_class::sirius_owned_s3, false, "S3"});
  REQUIRE_THROWS_WITH(
    require_cpu_replay(policy, "", "boom"),
    "S3 CPU fallback is not supported: this query reads s3:// data, GPU execution failed, and "
    "Sirius has no CPU fallback for S3 data sources. Underlying GPU error: boom");
  policy.scans = {
    {"sirius_stream_source", byte_source_class::stream, false, "stream source has no CPU body"}};
  REQUIRE_THROWS_WITH(
    require_cpu_replay(policy, "", mismatch_message),
    "CPU fallback is not supported: sirius_stream_source: stream source has no CPU body. "
    "Underlying GPU error: read-view mismatch: reason=no_correspondence, correspondence=none, "
    "original_count=0, candidate_count=0, different_total=0, different_original=0, "
    "different_candidate=0, "
    "original_hash=none, candidate_hash=none, original_depth=none, candidate_depth=none, "
    "only_original=[], only_candidate=[]");
  policy.scans.clear();
  policy.discovery_complete = false;
  REQUIRE_THROWS_WITH(
    require_cpu_replay(policy, "", "boom"),
    "CPU fallback is not supported: source discovery incomplete. Underlying GPU error: boom");
}

TEST_CASE("Source policy retains known S3 evidence after a sibling extraction failure",
          "[transparent][policy][shared_context]")
{
  REQUIRE(sirius::test::g_shared_env);
  auto con = sirius::test::g_shared_env->make_connection();
  duckdb::LogicalDummyScan root(0);
  auto broken = file_scan("broken", "local.parquet");
  broken->bind_data->Cast<duckdb::MultiFileBindData>().file_list.reset();
  root.children.push_back(std::move(broken));
  root.children.push_back(file_scan("remote_after_failure", "s3://bucket/data.parquet"));
  auto policy = derive_plan_source_policy(root, *con.context);
  REQUIRE_FALSE(policy.discovery_complete);
  REQUIRE_FALSE(policy.cpu_replay_permitted());
  REQUIRE(policy.reads_sirius_owned_s3());
  REQUIRE(policy.reason().find("remote_after_failure") != std::string::npos);
}

// These cases exercise the explicit entry point. Its window-entry failure happens
// after bind-time source discovery and before any execution-time plan extraction.
TEST_CASE("Explicit replay rejects bind-time source discovery failure before window entry",
          "[integration][policy][explicit_replay]")
{
  if (sirius::test::run_isolated()) return;
  sirius::test::GpuExecutionFixture fixture;
  auto& con = *fixture.con;
  fixture.run_ok("SET gpu_execution = false");
  fixture.run_ok("SET enable_duckdb_fallback = true");

  duckdb::ExtensionLoader loader(*con.context->db, "replay_policy_test");
  loader.RegisterFunction(duckdb::TableFunction(
    "unreadable_policy_source",
    {},
    [](duckdb::ClientContext&, duckdb::TableFunctionInput&, duckdb::DataChunk&) {
      throw std::runtime_error("unexpected CPU source execution");
    },
    [](duckdb::ClientContext&,
       duckdb::TableFunctionBindInput&,
       duckdb::vector<duckdb::LogicalType>& types,
       duckdb::vector<std::string>& names) -> duckdb::unique_ptr<duckdb::FunctionData> {
      types           = {duckdb::LogicalType::INTEGER};
      names           = {"id"};
      auto bind       = duckdb::make_uniq<duckdb::MultiFileBindData>();
      bind->types     = types;
      bind->names     = names;
      bind->file_list = duckdb::make_shared_ptr<throwing_list>();
      return std::move(bind);
    }));
  auto context = con.context->registered_state->Get<duckdb::SiriusContext>("sirius_state");
  REQUIRE(context);
  unsigned replays                     = 0;
  context->cpu_replay_hook_for_testing = [&] { ++replays; };
  struct reset_hook {
    duckdb::SiriusContext& context;
    ~reset_hook() { context.cpu_replay_hook_for_testing = {}; }
  } reset{*context};
  fixture.run_ok("SET sirius_test_sync_cpu_replay = true");
  fixture.run_ok("SET sirius_test_mark_runtime_unavailable_before_window = true");

  auto prepared =
    con.Prepare("SELECT * FROM gpu_execution('SELECT * FROM unreadable_policy_source()')");
  REQUIRE(prepared);
  INFO((prepared->HasError() ? prepared->GetError() : ""));
  REQUIRE_FALSE(prepared->HasError());
  REQUIRE(context->get_runtime_health() != duckdb::SiriusContext::runtime_health::UNAVAILABLE);
  auto result = prepared->Execute();
  REQUIRE(result);
  REQUIRE(result->HasError());
  INFO((result->HasError() ? result->GetError() : ""));
  CHECK(result->GetError().find("source discovery incomplete") != std::string::npos);
  CHECK(result->GetError().find("Sirius GPU runtime is unavailable") != std::string::npos);
  CHECK(result->GetError().find("unexpected CPU source execution") == std::string::npos);
  CHECK(replays == 0);
  CHECK(context->get_runtime_health() == duckdb::SiriusContext::runtime_health::UNAVAILABLE);
  CHECK_FALSE(context->get_scan_manager().holds_any_checkpoint_key());

  // Disabling fallback must keep the execution error ahead of the source veto.
  fixture.run_ok("SET enable_duckdb_fallback = false");
  result = con.Query("SELECT * FROM gpu_execution('SELECT * FROM unreadable_policy_source()')");
  REQUIRE(result->HasError());
  CHECK(result->GetError().find("SiriusExecuteQuery error:") != std::string::npos);
  CHECK(result->GetError().find("source discovery incomplete") == std::string::npos);
  CHECK(replays == 0);
}

TEST_CASE("Explicit replay permits local Parquet after refused window entry",
          "[integration][policy][explicit_replay]")
{
  if (sirius::test::run_isolated()) return;
  sirius::test::GpuExecutionFixture fixture;
  auto& con = *fixture.con;
  fixture.run_ok("SET gpu_execution = false");
  fixture.run_ok("SET enable_duckdb_fallback = true");
  auto path = std::filesystem::path(fixture.temp_db_path + ".parquet");
  struct remove_file {
    std::filesystem::path path;
    ~remove_file()
    {
      std::error_code error;
      std::filesystem::remove(path, error);
    }
  } cleanup{path};
  fixture.run_ok("COPY (SELECT 42::INTEGER AS id) TO '" + path.string() + "' (FORMAT PARQUET)");
  auto context = con.context->registered_state->Get<duckdb::SiriusContext>("sirius_state");
  REQUIRE(context);
  unsigned replays                     = 0;
  context->cpu_replay_hook_for_testing = [&] { ++replays; };
  struct reset_hook {
    duckdb::SiriusContext& context;
    ~reset_hook() { context.cpu_replay_hook_for_testing = {}; }
  } reset{*context};
  fixture.run_ok("SET sirius_test_sync_cpu_replay = true");
  fixture.run_ok("SET sirius_test_mark_runtime_unavailable_before_window = true");
  auto prepared = con.Prepare("SELECT * FROM gpu_execution('SELECT id FROM read_parquet(''" +
                              path.string() + "'')')");
  REQUIRE(prepared);
  INFO((prepared->HasError() ? prepared->GetError() : ""));
  REQUIRE_FALSE(prepared->HasError());
  REQUIRE(context->get_runtime_health() != duckdb::SiriusContext::runtime_health::UNAVAILABLE);
  auto result = prepared->Execute();
  REQUIRE(result);
  INFO((result->HasError() ? result->GetError() : ""));
  REQUIRE_FALSE(result->HasError());
  auto chunk = result->Fetch();
  REQUIRE(chunk);
  REQUIRE(chunk->size() == 1);
  CHECK(chunk->GetValue(0, 0).GetValue<int32_t>() == 42);
  auto end = result->Fetch();
  CHECK((!end || end->size() == 0));
  CHECK(replays == 1);
  CHECK(context->get_runtime_health() == duckdb::SiriusContext::runtime_health::UNAVAILABLE);
  CHECK_FALSE(context->get_scan_manager().holds_any_checkpoint_key());
}

namespace {
int precedence_bind_count = 0;
std::string precedence_replan_error;

duckdb::unique_ptr<duckdb::FunctionData> bind_precedence_source(
  duckdb::ClientContext&,
  duckdb::TableFunctionBindInput&,
  duckdb::vector<duckdb::LogicalType>& types,
  duckdb::vector<std::string>& names)
{
  if (++precedence_bind_count > 1) {
    if (precedence_replan_error == "unsupported")
      throw duckdb::NotImplementedException("source replan refused");
    if (precedence_replan_error == "unavailable")
      throw duckdb::SiriusRuntimeUnavailableException("source runtime unavailable");
    if (precedence_replan_error == "generic") throw std::runtime_error("source replan failed");
  }
  types           = {duckdb::LogicalType::INTEGER};
  names           = {"id"};
  auto bind       = duckdb::make_uniq<duckdb::MultiFileBindData>();
  bind->types     = types;
  bind->names     = names;
  bind->file_list = duckdb::make_shared_ptr<throwing_list>();
  return std::move(bind);
}
}  // namespace

TEST_CASE("Transparent finalize preserves GPU errors before incomplete-source veto",
          "[integration][policy][transparent_precedence]")
{
  sirius::test::GpuExecutionFixture fixture;
  auto& con = *fixture.con;
  duckdb::ExtensionLoader loader(*con.context->db, "precedence_test");
  loader.RegisterFunction(duckdb::TableFunction(
    "precedence_source",
    {},
    [](duckdb::ClientContext&, duckdb::TableFunctionInput&, duckdb::DataChunk&) {
      FAIL("CPU source must not execute");
    },
    bind_precedence_source));
  fixture.run_ok("SET gpu_execution = true");
  auto context = con.context->registered_state->Get<duckdb::SiriusContext>("sirius_state");
  REQUIRE(context);
  for (std::string replan_error : {"unsupported", "unavailable", "generic", ""}) {
    for (bool fallback : {false, true}) {
      CAPTURE(replan_error, fallback);
      fixture.run_ok(std::string("SET enable_duckdb_fallback = ") + (fallback ? "true" : "false"));
      precedence_bind_count   = 0;
      precedence_replan_error = replan_error;
      auto before             = context->get_transparent_execution_stats();
      auto result             = con.Query("SELECT * FROM precedence_source()");
      REQUIRE(result);
      REQUIRE(result->HasError());
      INFO(result->GetError());
      CHECK(precedence_bind_count >= 2);
      if (fallback) {
        CHECK(result->GetError().find(
                "CPU fallback is not supported: source discovery incomplete") != std::string::npos);
      } else {
        CHECK(result->GetError().find("CPU fallback is not supported") == std::string::npos);
        if (replan_error == "unavailable") {
          CHECK(result->GetErrorType() == duckdb::ExceptionType::EXECUTOR);
          CHECK(result->GetError().find("source runtime unavailable") != std::string::npos);
          // DuckDB converts the bind callback's derived exception to ExecutorException.
          CHECK(result->GetError().find("GPU plan generation failed:") != std::string::npos);
        } else {
          CHECK(result->GetErrorType() == (replan_error == "generic"
                                             ? duckdb::ExceptionType::INVALID
                                             : duckdb::ExceptionType::NOT_IMPLEMENTED));
          CHECK(result->GetError().find("GPU plan generation failed:") != std::string::npos);
          if (replan_error == "unsupported")
            CHECK(result->GetError().find("source replan refused") != std::string::npos);
          if (replan_error == "generic")
            CHECK(result->GetError().find("source replan failed") != std::string::npos);
        }
      }
      auto after = context->get_transparent_execution_stats();
      CHECK(after.fallbacks == before.fallbacks);
      CHECK(after.runtime_fallbacks == before.runtime_fallbacks);
    }
  }
}

TEST_CASE("Transparent stream planning preserves errors when fallback is disabled",
          "[integration][policy][transparent_precedence]")
{
  sirius::test::GpuExecutionFixture fixture;
  auto& con    = *fixture.con;
  auto catalog = sirius::exec::catalog_for(*con.context);
  fixture.run_ok("SET gpu_execution = true");
  for (bool fallback : {false, true}) {
    CAPTURE(fallback);
    fixture.run_ok(std::string("SET enable_duckdb_fallback = ") + (fallback ? "true" : "false"));
    catalog->declare(
      0,
      sirius::exec::stream_input_binding{
        {"id"},
        sirius::from_duckdb_vec(duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::INTEGER}),
        std::make_shared<cucascade::shared_data_repository>(),
        {0},
        nullptr});
    auto result = con.Query("SELECT row_number() OVER () FROM sirius_stream_source(0)");
    REQUIRE(result);
    REQUIRE(result->HasError());
    INFO(result->GetError());
    if (fallback) {
      CHECK(result->GetError().find("sirius_stream_source: stream source has no CPU body") !=
            std::string::npos);
    } else {
      CHECK(result->GetErrorType() == duckdb::ExceptionType::NOT_IMPLEMENTED);
      CHECK(result->GetError().find("GPU plan generation failed:") != std::string::npos);
      CHECK(result->GetError().find("CPU fallback is not supported") == std::string::npos);
    }
  }
}

TEST_CASE("Transparent execution preserves GPU errors before non-S3 replay vetoes",
          "[integration][policy][transparent_precedence]")
{
  if (sirius::test::run_isolated()) return;
  sirius::test::GpuExecutionFixture fixture;
  auto& con = *fixture.con;
  fixture.run_ok("SET gpu_execution = false");
  auto context = con.context->registered_state->Get<duckdb::SiriusContext>("sirius_state");
  REQUIRE(context);
  bool unavailable = false;
  SECTION("ordinary execution failure") {}
  SECTION("runtime unavailable")
  {
    unavailable = true;
    context->mark_runtime_unavailable();
  }
  for (auto const source : {"stream", "incomplete", "s3", "s3_text"}) {
    for (bool fallback : {false, true}) {
      CAPTURE(source, fallback, unavailable);
      fixture.run_ok(std::string("SET enable_duckdb_fallback = ") + (fallback ? "true" : "false"));
      plan_source_policy policy;
      std::string query;
      bool s3 = std::string_view(source).starts_with("s3");
      if (std::string_view(source) == "stream") {
        policy.scans.push_back({"sirius_stream_source",
                                byte_source_class::stream,
                                false,
                                "stream source has no CPU body"});
      } else if (std::string_view(source) == "incomplete") {
        policy.discovery_complete = false;
      } else if (std::string_view(source) == "s3") {
        policy.scans.push_back({"read_parquet", byte_source_class::sirius_owned_s3, false, "S3"});
      } else {
        // Fail parsing before any remote access; the SQL-text S3 veto must still win.
        query = "SELECT FROM read_parquet('s3://bucket/data.parquet')";
      }
      duckdb::PhysicalPlan plan(duckdb::Allocator::Get(*con.context));
      auto cpu = duckdb::make_shared_ptr<duckdb::PreparedStatementData>(
        duckdb::StatementType::SELECT_STATEMENT);
      // No GPU template: a deterministic execution error before materialization.
      // A non-null CPU stash keeps the test focused on the fallback switch.
      PhysicalSiriusExecution op(plan,
                                 nullptr,
                                 candidate_origin::copy,
                                 std::nullopt,
                                 {},
                                 query,
                                 {duckdb::LogicalType::INTEGER},
                                 {"id"},
                                 cpu,
                                 policy,
                                 0);
      duckdb::ThreadContext thread(*con.context);
      duckdb::ExecutionContext execution(*con.context, thread, nullptr);
      auto global = op.GetGlobalSourceState(*con.context);
      auto local  = op.GetLocalSourceState(execution, *global);
      duckdb::InterruptState interrupt;
      duckdb::OperatorSourceInput input{*global, *local, interrupt};
      duckdb::DataChunk output;
      duckdb::ErrorData error;
      try {
        op.GetDataInternal(execution, output, input);
      } catch (std::exception& e) {
        error = duckdb::ErrorData(e);
      }
      REQUIRE(error.HasError());
      INFO(error.RawMessage());
      if (unavailable && s3) {
        CHECK(error.Type() == duckdb::ExceptionType::EXECUTOR);
        CHECK(error.RawMessage().find("Sirius GPU runtime is unavailable") != std::string::npos);
        CHECK(error.RawMessage().find("CPU fallback is not supported") == std::string::npos);
      } else if (s3) {
        CHECK(error.RawMessage().find("S3 CPU fallback is not supported") != std::string::npos);
      } else if (!fallback) {
        CHECK(error.Type() == duckdb::ExceptionType::EXECUTOR);
        CHECK(error.RawMessage().starts_with("Sirius GPU execution failed: "));
        CHECK(error.RawMessage().find("CPU fallback is not supported") == std::string::npos);
        CHECK(error.RawMessage().find(unavailable ? "Sirius GPU runtime is unavailable"
                                                  : "missing the logical plan template") !=
              std::string::npos);
      } else {
        CHECK(error.RawMessage().find("CPU fallback is not supported:") != std::string::npos);
        CHECK(error.RawMessage().find(std::string_view(source) == "stream"
                                        ? "stream source has no CPU body"
                                        : "source discovery incomplete") != std::string::npos);
      }
      CHECK_FALSE(context->get_scan_manager().holds_any_checkpoint_key());
    }
  }
}
