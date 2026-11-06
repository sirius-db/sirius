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

#include "data/data_batch_utils.hpp"
#include "ducklake/binding_bridge.hpp"
#include "ducklake/dv_visibility.hpp"
#include "ducklake/inline_reader.hpp"
#include "ducklake/parquet_unit_ingestible.hpp"
#include "ducklake/scan_ingestible.hpp"
#include "exec/streaming_fragment.hpp"
#include "io/path_utils.hpp"
#include "io/s3/duckdb_secret_config.hpp"
#include "op/scan/arrow/arrow_import_ingestible.hpp"
#include "op/scan/puffin_io.hpp"
#include "op/scan/puffin_reader.hpp"
#include "planner/connector_registry.hpp"
#include "scan_manager/preparation_fixture.hpp"
#include "scan_manager/preparation_test_support.hpp"
#include "transparent/connection_provenance.hpp"
#include "util/env_guard.hpp"
#include "utils/dynamic_filter_test_utils.hpp"
#include "utils/offset_fault_server.hpp"
#include "utils/s3_backend.hpp"
#include "utils/s3_test_env.hpp"
#include "utils/scoped_temp_directory.hpp"
#include "utils/sirius_test_env.hpp"
#include "utils/transparent_execution_test_utils.hpp"

#include <cuda_runtime_api.h>

#include <catch.hpp>
#include <cucascade/cudf/datasource.hpp>
#include <cucascade/io/io_errors.hpp>
#include <cucascade/memory/memory_space.hpp>
#include <duckdb/common/enums/statement_type.hpp>
#include <duckdb/main/attached_database.hpp>
#include <duckdb/main/client_config.hpp>
#include <duckdb/main/client_context_state.hpp>
#include <duckdb/main/config.hpp>
#include <duckdb/main/database_manager.hpp>
#include <duckdb/main/extension_helper.hpp>
#include <duckdb/optimizer/optimizer_extension.hpp>
#include <duckdb/planner/operator/logical_get.hpp>
#include <loopback_range_server.hpp>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <future>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <utility>

namespace {
using sirius::test::get_transparent_execution_stats;
using sirius::test::query;
using sirius::transparent::connection_provenance;
using sirius::transparent::decline_reason;

bool load_ducklake(duckdb::Connection& con)
{
  auto path = std::getenv("SIRIUS_TEST_DUCKLAKE_EXTENSION");
  auto result =
    con.Query(path && *path ? "LOAD " + duckdb::Value(path).ToSQLString() : "LOAD ducklake");
  if (!result || result->HasError()) {
    auto required = std::getenv("SIRIUS_TEST_REQUIRE_DUCKLAKE");
    if (required && std::string(required) == "1") {
      FAIL("required DuckLake LOAD failed: " << (result ? result->GetError() : "no result"));
    }
    WARN("DuckLake not covered");
    return false;
  }
  auto version = query(con,
                       "SELECT extension_version FROM duckdb_extensions() WHERE extension_name = "
                       "'ducklake' AND loaded");
  REQUIRE(version->RowCount() == 1);
  REQUIRE(version->Collection().GetValue(0, 0).ToString() == "417a9117f5");
  return true;
}

void attach_lake(duckdb::Connection& con, std::string const& directory)
{
  query(con,
        "ATTACH 'ducklake:" + directory + "/meta.ducklake' AS lake (DATA_PATH '" + directory +
          "/data/')");
}

void populate_lake(duckdb::Connection& con)
{
  query(con, "CREATE TABLE lake.t(id INTEGER)");
  query(con, "INSERT INTO lake.t SELECT i FROM range(100) r(i)");
  query(con, "INSERT INTO lake.t VALUES (100), (101)");
  query(con, "DELETE FROM lake.t WHERE id = 5");
  query(con, "ALTER TABLE lake.t ADD COLUMN extra INTEGER DEFAULT 7");
  query(con, "INSERT INTO lake.t SELECT i, 7 FROM range(102, 132) r(i)");
  query(con, "CALL ducklake_flush_inlined_data('lake')");
}

void check_lake_rows(duckdb::Connection& con)
{
  auto rows = query(con, "SELECT id, extra FROM lake.t ORDER BY id");
  REQUIRE(rows->RowCount() == 131);
  for (int row = 0; row < 131; ++row) {
    REQUIRE(rows->Collection().GetValue(0, row).GetValue<int32_t>() == (row < 5 ? row : row + 1));
    REQUIRE(rows->Collection().GetValue(1, row).GetValue<int32_t>() == 7);
  }
  auto aggregates = query(con, "SELECT count(*), sum(id), sum(extra) FROM lake.t");
  REQUIRE(aggregates->Collection().GetValue(0, 0).GetValue<int64_t>() == 131);
  REQUIRE(aggregates->Collection().GetValue(1, 0).GetValue<int64_t>() == 8641);
  REQUIRE(aggregates->Collection().GetValue(2, 0).GetValue<int64_t>() == 917);
}

struct provider_query_observer : duckdb::OptimizerExtensionInfo {
  std::function<void(duckdb::ClientContext&)> observe;
  std::function<void(duckdb::ClientContext&, duckdb::LogicalOperator&)> inspect_plan;
};
void observe_provider_query(duckdb::OptimizerExtensionInput& input,
                            duckdb::unique_ptr<duckdb::LogicalOperator>& plan)
{
  auto& observer = static_cast<provider_query_observer&>(*input.info);
  if (observer.observe) observer.observe(input.context);
  if (observer.inspect_plan) observer.inspect_plan(input.context, *plan);
}

duckdb::LogicalGet* ducklake_get(duckdb::LogicalOperator& plan)
{
  if (plan.type == duckdb::LogicalOperatorType::LOGICAL_GET) {
    auto& get = plan.Cast<duckdb::LogicalGet>();
    if (get.function.GetName() == "ducklake_scan") return &get;
  }
  for (auto& child : plan.children)
    if (auto* get = ducklake_get(*child)) return get;
  return nullptr;
}

auto lake_config() { return sirius::test::integration_config_path(); }

template <typename Configure>
auto write_lake_config(std::filesystem::path const& directory, Configure configure)
{
  auto config = YAML::LoadFile(lake_config().string());
  configure(config);
  auto path = directory / "config.yaml";
  std::ofstream output(path);
  output << config;
  REQUIRE(output.good());
  return path;
}

struct query_pause {
  std::promise<void> arrival, continuation;
  std::future<void> arrived       = arrival.get_future();
  std::shared_future<void> resume = continuation.get_future().share();
  bool released                   = false;

  void arrive_and_wait(std::chrono::seconds timeout = std::chrono::seconds(15))
  {
    arrival.set_value();
    if (resume.wait_for(timeout) != std::future_status::ready)
      throw std::runtime_error("query pause timed out");
  }
  bool wait_until_arrived()
  {
    return arrived.wait_for(std::chrono::seconds(15)) == std::future_status::ready;
  }
  void release()
  {
    if (!std::exchange(released, true)) continuation.set_value();
  }
};

std::vector<std::string> ducklake_result_bag(duckdb::QueryResult& result)
{
  if (result.HasError()) throw std::runtime_error(result.GetError());
  std::vector<std::string> rows;
  while (auto chunk = result.Fetch()) {
    for (duckdb::idx_t row = 0; row < chunk->size(); ++row) {
      std::string value;
      for (duckdb::idx_t column = 0; column < chunk->ColumnCount(); ++column)
        value += chunk->GetValue(column, row).ToString() + "|";
      rows.push_back(std::move(value));
    }
  }
  std::sort(rows.begin(), rows.end());
  return rows;
}

// Observe real HOST backing without replacing its reservation or allocation policy.
struct observed_host_reservations : sirius::scan_manager::reservation_provider {
  struct observations {
    std::mutex mutex;
    uint64_t grants = 0, live = 0, peak_live = 0, peak_grants = 0, host_capacity = 0;
    size_t active = 0, peak_active = 0;
    bool exceeded = false;
    std::function<void(uint64_t)> on_allocate;
  };
  struct backing : sirius::scan_manager::reservation_backing,
                   std::enable_shared_from_this<backing> {
    std::shared_ptr<sirius::scan_manager::reservation_backing> original;
    std::shared_ptr<observations> seen;
    uint64_t capacity, bound, live = 0;
    backing(std::shared_ptr<sirius::scan_manager::reservation_backing> source,
            std::shared_ptr<observations> stats,
            uint64_t bytes)
      : original(std::move(source)), seen(std::move(stats)), capacity(bytes), bound(bytes)
    {
      std::lock_guard lock(seen->mutex);
      seen->grants += capacity;
      ++seen->active;
      seen->peak_grants = std::max(seen->peak_grants, seen->grants);
      seen->peak_active = std::max(seen->peak_active, seen->active);
      seen->exceeded |= seen->grants > seen->host_capacity;
    }
    ~backing() override
    {
      original.reset();
      std::lock_guard lock(seen->mutex);
      seen->grants -= capacity;
      --seen->active;
    }
    uint64_t charge_size(uint64_t bytes) const override { return original->charge_size(bytes); }
    sirius::scan_manager::backing_allocation allocate(uint64_t bytes) override
    {
      auto allocation = original->allocate(bytes);
      if (seen->on_allocate) seen->on_allocate(bytes);
      auto charge = charge_size(bytes);
      {
        std::lock_guard lock(seen->mutex);
        live += charge;
        seen->live += charge;
        seen->peak_live = std::max(seen->peak_live, seen->live);
        seen->exceeded |= live > bound || seen->live > seen->grants;
      }
      auto owner = std::shared_ptr<void>(
        allocation.data,
        [self = shared_from_this(), held = std::move(allocation.owner), charge](void*) mutable {
          held.reset();
          std::lock_guard lock(self->seen->mutex);
          self->live -= charge;
          self->seen->live -= charge;
        });
      return {allocation.data, std::move(owner)};
    }
  };
  sirius::scan_manager::memory_space_id space{cucascade::memory::Tier::HOST, 0};
  std::map<std::thread::id, std::weak_ptr<backing>> pending;
  sirius::scan_manager::host_reservation_provider original;
  std::shared_ptr<observations> seen = std::make_shared<observations>();
  explicit observed_host_reservations(cucascade::memory::memory_reservation_manager& manager)
    : original(manager)
  {
    auto hosts = manager.get_memory_spaces_for_tier(cucascade::memory::Tier::HOST);
    if (hosts.size() != 1) throw std::runtime_error("HOST observer requires one shared space");
    seen->host_capacity = hosts.front()->get_max_memory();
    space               = hosts.front()->get_id();
  }
  uint64_t allocation_granularity(sirius::scan_manager::memory_space_id space) const override
  {
    return original.allocation_granularity(space);
  }
  std::optional<sirius::scan_manager::reservation_grant> request(
    sirius::scan_manager::memory_space_id space, uint64_t bytes) override
  {
    auto grant = original.request(space, bytes);
    if (grant) {
      auto tracked  = std::make_shared<backing>(std::move(grant->handle), seen, grant->bytes);
      grant->handle = tracked;
      std::lock_guard lock(seen->mutex);
      pending[std::this_thread::get_id()] = tracked;
    }
    return grant;
  }
  void record_admission(sirius::scan_manager::admission_decision const& decision,
                        unsigned minimum,
                        unsigned ceiling)
  {
    std::lock_guard lock(seen->mutex);
    auto tracked = pending[std::this_thread::get_id()].lock();
    if (!decision.deferred || !tracked) throw std::runtime_error("missing HOST grant observation");
    tracked->bound = decision.sigma_retained + decision.n_permits * decision.w;
    seen->exceeded |= tracked->live > tracked->bound || tracked->bound > tracked->capacity ||
                      tracked->capacity != decision.c_obtained || decision.n_permits < minimum ||
                      decision.n_permits > ceiling;
    pending.erase(std::this_thread::get_id());
  }
};

// Exercise actual provider children with spare ledger capacity, so pending proves gate exclusion.
void check_mixed_input_gate(std::shared_ptr<sirius::ducklake::binding_handle> const& binding,
                            observed_host_reservations& backing,
                            std::string const& deletion)
{
  namespace dl = sirius::ducklake;
  namespace sm = sirius::scan_manager;
  auto verify  = [](bool ok) {
    if (!ok) throw std::runtime_error("DuckLake mixed input gate invariant failed");
  };
  binding->bind_execution_owner();
  binding->configure_scan({0}, nullptr);
  std::optional<uint64_t> parquet_index, inline_index;
  for (uint64_t index = 0; auto unit = binding->unit_at(index); ++index) {
    if (unit->kind == dl::unit_kind::parquet) {
      verify(deletion == "inline"
               ? unit->inline_delete_count > 0
               : unit->deletes ==
                   (deletion == "puffin" ? dl::delete_format::puffin : dl::delete_format::parquet));
      parquet_index = index;
    } else
      inline_index = index;
  }
  verify(parquet_index && inline_index);
  auto envelope = dl::scan_ingestible::envelope(1, 0);
  std::vector<sm::scan_envelope> envelopes{{{envelope, envelope}, {}, true, 1}};
  auto unique   = std::make_unique<sm::preparation_ledger>(backing);
  auto decision = unique->admit(envelopes, backing.space, 2);
  backing.record_admission(decision, 2, 2);
  verify(decision.n_permits == 2);
  auto admission  = std::make_shared<sm::preparation_admission>(std::move(unique), decision);
  auto ledger     = std::shared_ptr<sm::preparation_ledger>(admission->consume());
  auto identities = std::make_shared<dl::preparation_identity>();
  auto gate       = std::make_shared<dl::preparation_gate>();
  auto slot       = std::make_shared<dl::preparation_slot>(
    ledger, sm::unit_key{1, identities->next()}, envelope.retained);
  auto options = [&] {
    dl::unit_scan_options result;
    result.identities     = identities;
    result.inline_gate    = gate;
    result.parquet_slot   = slot;
    result.owns_admission = false;
    return result;
  };
  auto parquet = binding->make_parquet_unit(*parquet_index, 1, 1, admission, {0}, false, options());
  auto arrow   = binding->make_inline_unit(*inline_index, 1, 1, admission, false, {0}, options());
  parquet->attach_preparation_ledger(ledger);
  arrow->attach_preparation_ledger(ledger);
  auto finish = std::shared_ptr<void>(nullptr, [&](void*) {
    arrow->finish_preparation();
    parquet->finish_preparation();
    ledger->close();
  });
  parquet->preflight_for_query();
  arrow->preflight_for_query();
  auto sources = parquet->make_preparation_sources(
    [](std::string_view) -> std::shared_ptr<cucascade::io::ioctx> {
      throw std::runtime_error("abandoned gate probe must not start Parquet I/O");
    });
  auto claim = arrow->try_claim_split({});
  verify(std::holds_alternative<sm::preparation_job>(claim));
  auto input = std::get<sm::preparation_job>(std::move(claim)).run({});
  verify(input && gate->active && ledger->permits_in_flight() == 1);
  auto extra = dynamic_cast<sirius::op::scan::owned_arrow_scan_info&>(*input).coalesced_copy();
  input.reset();
  verify(std::holds_alternative<sm::preparation_pending>(sources.front().try_claim()));
  extra.reset();
  verify(!gate->active && ledger->permits_in_flight() == 0);
  {
    auto delete_claim = sources.front().try_claim();
    verify(std::holds_alternative<sm::preparation_job>(delete_claim));
    verify(gate->active && ledger->permits_in_flight() == 1);
    verify(std::holds_alternative<sm::preparation_pending>(arrow->try_claim_split({})));
    // Abandon the delete job before I/O; its owner must return the same gate and permit.
  }
  verify(!gate->active && ledger->permits_in_flight() == 0);
  auto next = arrow->try_claim_split({});
  verify(std::holds_alternative<sm::preparation_job>(next));
  std::get<sm::preparation_job>(std::move(next)).run({});
}

struct ducklake_fixture {
  sirius::test::scoped_temp_directory disk;
  sirius::test::shared_test_env env;
  duckdb::Connection con{env.make_connection()};

  explicit ducklake_fixture(std::filesystem::path const& config = lake_config()) : env(config)
  {
    if (!load_ducklake(con)) SKIP("DuckLake extension is unavailable");
  }

  auto run(std::string const& sql) { return query(con, sql); }

  void create_recovery_table(int files, bool flush_after_insert = true)
  {
    attach_lake(con, disk.directory);
    run("SET gpu_execution=false");
    run("CALL lake.set_option('data_inlining_row_limit',100)");
    run("CREATE TABLE lake.q(id BIGINT,value BIGINT)");
    for (int file = 0; file < files; ++file) {
      run("INSERT INTO lake.q SELECT i,i%7 FROM range(" + std::to_string(file * 4096) + "," +
          std::to_string((file + 1) * 4096) + ") r(i)");
      if (flush_after_insert) run("CALL ducklake_flush_inlined_data('lake')");
    }
    run("INSERT INTO lake.q VALUES (20000,5),(20000,5),(20001,NULL)");
  }

  auto cpu_result(std::string const& sql)
  {
    run("SET gpu_execution=false");
    auto expected = ducklake_result_bag(*run(sql));
    run("SET gpu_execution=true");
    return expected;
  }

  template <typename Execute>
  void check_gpu_execution(std::vector<std::string> const& expected,
                           Execute execute,
                           bool explicit_entry = false)
  {
    CAPTURE(explicit_entry);
    run("SET gpu_execution=true");
    run("SET enable_duckdb_fallback=false");
    auto runtime = sirius::test::get_registered_sirius_context(con);
    auto hosts =
      runtime->get_memory_manager().get_memory_spaces_for_tier(cucascade::memory::Tier::HOST);
    REQUIRE(hosts.size() == 1);
    auto reserved = hosts[0]->get_total_reserved_memory();
    auto before   = get_transparent_execution_stats(con);
    auto result   = execute();
    INFO((result->HasError() ? result->GetError() : "GPU query completed"));
    REQUIRE_FALSE(result->HasError());
    CHECK(ducklake_result_bag(*result) == expected);
    result.reset();
    auto after = get_transparent_execution_stats(con);
    if (explicit_entry) {
      CHECK(after.executions == before.executions);
      CHECK(after.fallbacks == before.fallbacks);
      CHECK(after.runtime_fallbacks == before.runtime_fallbacks);
      CHECK(after.certificate_incompletes == before.certificate_incompletes);
      CHECK(after.certificate_mismatches == before.certificate_mismatches);
    } else {
      sirius::test::require_transparent_execution_delta(before, after, 1, 0, 1);
    }
    CHECK(hosts[0]->get_total_reserved_memory() == reserved);
  }

  void check_gpu_result(std::string const& sql,
                        std::vector<std::string> const& expected,
                        bool explicit_entry = false)
  {
    CAPTURE(sql);
    check_gpu_execution(
      expected,
      [&] {
        return con.Query(
          explicit_entry ? "CALL gpu_execution(" + duckdb::Value(sql).ToSQLString() + ")" : sql);
      },
      explicit_entry);
  }

  void check_gpu_query(std::string const& sql) { check_gpu_result(sql, cpu_result(sql)); }
};

}  // namespace

TEST_CASE("DuckLake provider SQL declines while ordinary user SQL remains GPU eligible",
          "[transparent][provenance][ducklake][isolated_context]")
{
  sirius::test::scoped_temp_directory disk;
  sirius::test::shared_test_env env(lake_config());
  auto con = env.make_connection();
  if (!load_ducklake(con)) { return; }
  query(con, "SET GLOBAL gpu_execution = true");
  auto before = get_transparent_execution_stats(con);
  attach_lake(con, disk.directory);
  populate_lake(con);
  query(con, "SELECT id, extra FROM lake.t ORDER BY id");
  auto after = get_transparent_execution_stats(con);
  REQUIRE(after.provider_internal_skips > before.provider_internal_skips);
  auto expected_executions = sirius::ducklake::bridge_available() ? 1 : 0;
  REQUIRE(after.successful_rebinds == before.successful_rebinds + expected_executions);
  REQUIRE(after.executions == before.executions + expected_executions);
  REQUIRE(after.classification_failures == before.classification_failures);
  before = after;
  query(con, "SELECT 42::INTEGER");
  after = get_transparent_execution_stats(con);
  sirius::test::require_transparent_execution_delta(before, after, 1, 0, 1);
  check_lake_rows(con);
}

TEST_CASE("DuckLake flush preserves rows and file paths after closing every connection",
          "[transparent][provenance][ducklake][isolated_context]")
{
  sirius::test::scoped_temp_directory disk;
  {
    sirius::test::shared_test_env env(lake_config());
    auto con = env.make_connection();
    if (!load_ducklake(con)) { return; }
    query(con, "SET GLOBAL gpu_execution = true");
    attach_lake(con, disk.directory);
    populate_lake(con);
    check_lake_rows(con);
    query(con, "DETACH lake");
  }
  {
    sirius::test::shared_test_env env(lake_config());
    auto con = env.make_connection();
    REQUIRE(load_ducklake(con));
    query(con, "SET GLOBAL gpu_execution = true");
    attach_lake(con, disk.directory);
    check_lake_rows(con);
    auto files = query(con, "SELECT data_file FROM ducklake_list_files('lake', 't')");
    REQUIRE(files->RowCount() > 0);
    for (duckdb::idx_t row = 0; row < files->RowCount(); ++row) {
      REQUIRE_FALSE(files->Collection().GetValue(0, row).IsNull());
      REQUIRE(std::filesystem::is_regular_file(files->Collection().GetValue(0, row).ToString()));
    }
    query(con, "DETACH lake");
    query(con, "ATTACH '" + disk.directory + "/meta.ducklake' AS meta (READ_ONLY)");
    query(con, "SET SESSION gpu_execution = false");
    auto paths = query(
      con,
      "SELECT f.path, f.path_is_relative, t.path, t.path_is_relative, s.path, s.path_is_relative "
      "FROM meta.ducklake_data_file f JOIN meta.ducklake_table t USING (table_id) "
      "JOIN meta.ducklake_schema s USING (schema_id) "
      "WHERE t.end_snapshot IS NULL AND s.end_snapshot IS NULL");
    REQUIRE(paths->RowCount() > 0);
    for (duckdb::idx_t row = 0; row < paths->RowCount(); ++row) {
      REQUIRE_FALSE(paths->Collection().GetValue(0, row).IsNull());
      auto path = std::filesystem::path(paths->Collection().GetValue(0, row).ToString());
      if (paths->Collection().GetValue(1, row).GetValue<bool>()) {
        auto table_path = std::filesystem::path(paths->Collection().GetValue(2, row).ToString());
        if (paths->Collection().GetValue(3, row).GetValue<bool>()) {
          auto schema_path = std::filesystem::path(paths->Collection().GetValue(4, row).ToString());
          if (paths->Collection().GetValue(5, row).GetValue<bool>()) {
            schema_path = std::filesystem::path(disk.directory) / "data" / schema_path;
          }
          table_path = schema_path / table_path;
        }
        path = table_path / path;
      }
      REQUIRE(std::filesystem::is_regular_file(path));
    }
  }
}

TEST_CASE_METHOD(ducklake_fixture,
                 "DuckLake borrowed bindings refuse access after their statement lifetime",
                 "[isolated_context][ducklake][ducklake_binding_lifetime]")
{
  auto end = GENERATE(std::string("statement"), std::string("rollback"), std::string("context"));
  CAPTURE(end);
  if (!sirius::ducklake::bridge_available()) SKIP("DuckLake private binding bridge is disabled");
  run("SET gpu_execution=false");
  attach_lake(con, disk.directory);
  run("CREATE TABLE lake.q AS SELECT 42::BIGINT id");
  auto reader = std::make_unique<duckdb::Connection>(env.database());
  query(*reader, "SET gpu_execution=false");
  query(*reader, "BEGIN");
  std::shared_ptr<sirius::ducklake::binding_handle> captured;
  auto observer          = duckdb::make_shared_ptr<provider_query_observer>();
  observer->inspect_plan = [&](duckdb::ClientContext& context, duckdb::LogicalOperator& plan) {
    if (&context != reader->context.get() || captured) return;
    auto visit = [&](auto&& self, duckdb::LogicalOperator& op) -> void {
      if (op.type == duckdb::LogicalOperatorType::LOGICAL_GET) {
        auto& get = op.Cast<duckdb::LogicalGet>();
        if (get.function.GetName() == "ducklake_scan") {
          captured =
            sirius::ducklake::binding_handle::capture(get.function, get.bind_data.get(), context);
          captured->bind_execution_owner();
          auto unit = captured->unit_at(0);
          REQUIRE(unit);
          REQUIRE(unit->kind != sirius::ducklake::unit_kind::parquet);
          auto cursor = captured->open_inline_unit(0, {0});
          size_t rows = 0;
          while (auto chunk = cursor->next())
            rows += chunk->size();
          CHECK(rows == 1);
          cursor.reset();
          captured->validate_statement();
        }
      }
      for (auto& child : op.children)
        self(self, *child);
    };
    visit(visit, plan);
  };
  duckdb::OptimizerExtension hook;
  hook.optimizer_info        = observer;
  hook.pre_optimize_function = observe_provider_query;
  duckdb::OptimizerExtension::Register(duckdb::DBConfig::GetConfig(*reader->context),
                                       std::move(hook));
  auto clear = std::shared_ptr<void>(nullptr, [&](void*) { observer->inspect_plan = {}; });
  CHECK(query(*reader, "SELECT * FROM lake.q")->RowCount() == 1);
  REQUIRE(captured);
  observer->inspect_plan = {};
  if (end == "rollback") query(*reader, "ROLLBACK");
  if (end == "context") reader.reset();
  auto refused = Catch::Matchers::ContainsSubstring("outside its captured statement/transaction");
  CHECK_THROWS_WITH(captured->validate_statement(), refused);
  CHECK_THROWS_WITH(captured->bind_execution_owner(), refused);
  CHECK_THROWS_WITH(captured->unit_at(0), refused);
  CHECK_THROWS_WITH(captured->open_inline_unit(0, {0}), refused);
}

TEST_CASE_METHOD(ducklake_fixture,
                 "DuckLake execution owner transfers only before provider discovery",
                 "[isolated_context][ducklake][ducklake_owner_transfer]")
{
  auto discovered = GENERATE(false, true);
  CAPTURE(discovered);
  if (!sirius::ducklake::bridge_available()) SKIP("DuckLake private binding bridge is disabled");
  run("SET gpu_execution=false");
  attach_lake(con, disk.directory);
  run("CREATE TABLE lake.q AS SELECT 42::BIGINT id");
  bool inspected         = false;
  auto observer          = duckdb::make_shared_ptr<provider_query_observer>();
  observer->inspect_plan = [&](duckdb::ClientContext& context, duckdb::LogicalOperator& plan) {
    if (&context != con.context.get() || inspected) return;
    auto visit = [&](auto&& self, duckdb::LogicalOperator& op) -> void {
      if (op.type == duckdb::LogicalOperatorType::LOGICAL_GET) {
        auto& get = op.Cast<duckdb::LogicalGet>();
        if (get.function.GetName() == "ducklake_scan") {
          inspected = true;
          auto binding =
            sirius::ducklake::binding_handle::capture(get.function, get.bind_data.get(), context);
          if (discovered) REQUIRE(binding->unit_at(0));
          // The planning owner stays idle until the execution owner has closed its reader.
          auto execution = std::async(std::launch::async, [&] {
            binding->bind_execution_owner();
            binding->bind_execution_owner();
            auto cursor = binding->open_inline_unit(0, {0});
            size_t rows = 0;
            while (auto chunk = cursor->next())
              rows += chunk->size();
            return rows;
          });
          auto refused   = Catch::Matchers::ContainsSubstring("statement owner thread");
          if (discovered) {
            CHECK_THROWS_WITH(execution.get(), refused);
            binding->validate_statement();
          } else {
            CHECK(execution.get() == 1);
            CHECK_THROWS_WITH(binding->validate_statement(), refused);
            CHECK_THROWS_WITH(binding->bind_execution_owner(), refused);
            CHECK_THROWS_WITH(binding->unit_at(0), refused);
          }
        }
      }
      for (auto& child : op.children)
        self(self, *child);
    };
    visit(visit, plan);
  };
  duckdb::OptimizerExtension hook;
  hook.optimizer_info        = observer;
  hook.pre_optimize_function = observe_provider_query;
  duckdb::OptimizerExtension::Register(duckdb::DBConfig::GetConfig(*con.context), std::move(hook));
  auto clear = std::shared_ptr<void>(nullptr, [&](void*) { observer->inspect_plan = {}; });
  CHECK(run("SELECT * FROM lake.q")->RowCount() == 1);
  CHECK(inspected);
}

TEST_CASE_METHOD(ducklake_fixture,
                 "DuckLake bound deletion locality follows the selected read view",
                 "[isolated_context][ducklake][ducklake_delete_locality]")
{
  if (!sirius::ducklake::bridge_available()) SKIP("DuckLake private binding bridge is disabled");
  query(con, "SET gpu_execution=false");
  attach_lake(con, disk.directory);
  query(con, "CALL lake.set_option('data_inlining_row_limit',10)");
  query(con, "CREATE TABLE lake.t AS SELECT i::BIGINT id FROM range(100) r(i)");
  query(con, "DELETE FROM lake.t WHERE id=0");
  auto snapshot  = query(con,
                        "SELECT snapshot_id, snapshot_time FROM ducklake_snapshots('lake') "
                         "ORDER BY snapshot_id DESC LIMIT 1");
  auto version   = snapshot->Collection().GetValue(0, 0).ToString();
  auto timestamp = snapshot->Collection().GetValue(1, 0).ToSQLString();
  query(con, "DELETE FROM lake.t WHERE id=2");
  query(con, "CALL ducklake_flush_inlined_data('lake')");
  query(con, "BEGIN");
  query(con, "DELETE FROM lake.t WHERE id>=50 AND id<80");

  auto observer = duckdb::make_shared_ptr<provider_query_observer>();
  std::vector<bool> local_deletes;
  bool inside            = false;
  observer->inspect_plan = [&](duckdb::ClientContext& context, duckdb::LogicalOperator& plan) {
    if (&context != con.context.get() || inside) return;
    inside     = true;
    auto leave = std::shared_ptr<void>(nullptr, [&](void*) { inside = false; });
    auto visit = [&](auto&& self, duckdb::LogicalOperator& op) -> void {
      if (op.type == duckdb::LogicalOperatorType::LOGICAL_GET) {
        auto& get = op.Cast<duckdb::LogicalGet>();
        if (get.function.GetName() == "ducklake_scan") {
          auto binding =
            sirius::ducklake::binding_handle::capture(get.function, get.bind_data.get(), context);
          for (uint64_t index = 0;; ++index) {
            auto unit = binding->unit_at(index);
            if (!unit) break;
            if (unit->kind != sirius::ducklake::unit_kind::parquet) continue;
            local_deletes.push_back(unit->transaction_local_delete);
          }
        }
      }
      for (auto& child : op.children)
        self(self, *child);
    };
    visit(visit, plan);
  };
  duckdb::OptimizerExtension hook;
  hook.optimizer_info        = observer;
  hook.pre_optimize_function = observe_provider_query;
  duckdb::OptimizerExtension::Register(duckdb::DBConfig::GetConfig(*con.context), std::move(hook));
  auto clear = std::shared_ptr<void>(nullptr, [&](void*) { observer->inspect_plan = {}; });
  auto check = [&](std::string const& suffix, bool expected_local, int64_t count, int64_t sum) {
    CAPTURE(suffix, expected_local);
    local_deletes.clear();
    auto result = query(con, "SELECT count(*), sum(id)::BIGINT FROM lake.t" + suffix);
    REQUIRE_FALSE(local_deletes.empty());
    for (auto local : local_deletes)
      CHECK(local == expected_local);
    CHECK(result->Collection().GetValue(0, 0).GetValue<int64_t>() == count);
    CHECK(result->Collection().GetValue(1, 0).GetValue<int64_t>() == sum);
  };
  check("", true, 68, 3013);
  check(" AT (VERSION => " + version + ")", false, 99, 4950);
  check(" AT (TIMESTAMP => " + timestamp + ")", false, 99, 4950);
  check("", true, 68, 3013);
  query(con, "ROLLBACK");
  check("", false, 98, 4948);
}

TEST_CASE_METHOD(ducklake_fixture,
                 "DuckLake SQL preserves snapshot and transaction visibility",
                 "[transparent][provenance][ducklake][isolated_context]")
{
  query(con, "SET GLOBAL gpu_execution = true");
  attach_lake(con, disk.directory);
  auto inline_limit = GENERATE(0, 1000);
  query(con,
        "CALL lake.set_option('data_inlining_row_limit', " + std::to_string(inline_limit) + ")");
  query(con, "CREATE TABLE lake.t(id BIGINT)");
  query(con, "INSERT INTO lake.t VALUES (1), (2)");
  auto snapshot  = query(con,
                        "SELECT snapshot_id, snapshot_time FROM ducklake_snapshots('lake') ORDER "
                         "BY snapshot_id DESC LIMIT 1");
  auto version   = snapshot->Collection().GetValue(0, 0).ToString();
  auto timestamp = snapshot->Collection().GetValue(1, 0).ToSQLString();
  query(con, "INSERT INTO lake.t VALUES (3)");
  query(con, "BEGIN");
  query(con, "INSERT INTO lake.t VALUES (4)");
  query(con, "DELETE FROM lake.t WHERE id = 1");
  auto before  = get_transparent_execution_stats(con);
  auto current = query(con, "SELECT sum(id)::BIGINT FROM lake.t");
  REQUIRE(current->Collection().GetValue(0, 0).GetValue<int64_t>() == 9);
  for (auto const& selector : {"VERSION => " + version, "TIMESTAMP => " + timestamp}) {
    // Explicit snapshots exclude this transaction's uncommitted inserts and deletes.
    auto sql = "SELECT sum(id)::BIGINT FROM lake.t AT (" + selector + ")";
    query(con, "SET SESSION gpu_execution = false");
    auto baseline = query(con, sql);
    REQUIRE(baseline->Collection().GetValue(0, 0).GetValue<int64_t>() == 3);
    query(con, "SET SESSION gpu_execution = true");
    auto selected = query(con, sql);
    REQUIRE(selected->Collection().GetValue(0, 0) == baseline->Collection().GetValue(0, 0));
  }
  auto prepared = con.Prepare("SELECT sum(id)::BIGINT FROM lake.t WHERE id >= ?");
  REQUIRE_FALSE(prepared->HasError());
  for (auto threshold : {2, 3}) {
    auto result = prepared->Execute(threshold);
    REQUIRE_FALSE(result->HasError());
    REQUIRE(result->Collection().GetValue(0, 0).GetValue<int64_t>() == (threshold == 2 ? 9 : 7));
  }
  auto after               = get_transparent_execution_stats(con);
  auto expected_executions = sirius::ducklake::bridge_available() ? 5 : 0;
  REQUIRE(after.executions == before.executions + expected_executions);
  REQUIRE(after.successful_rebinds == before.successful_rebinds + expected_executions);
  query(con, "ROLLBACK");
  auto committed = query(con, "SELECT sum(id)::BIGINT FROM lake.t");
  REQUIRE(committed->Collection().GetValue(0, 0).GetValue<int64_t>() == 6);
}

TEST_CASE_METHOD(ducklake_fixture,
                 "DuckLake unsupported columns respect explicit and transparent fallback settings",
                 "[transparent][provenance][ducklake][isolated_context]")
{
  query(con, "SET GLOBAL gpu_execution = true");
  attach_lake(con, disk.directory);
  query(con, "CREATE TABLE lake.t(id UUID)");
  auto value = std::string("550e8400-e29b-41d4-a716-446655440000");
  query(con, "INSERT INTO lake.t VALUES ('" + value + "')");
  auto sql          = std::string("SELECT id FROM lake.t");
  auto explicit_sql = "CALL gpu_execution(" + duckdb::Value(sql).ToSQLString() + ")";
  for (auto const& statement : {sql, explicit_sql}) {
    query(con, "SET enable_duckdb_fallback = true");
    auto result = query(con, statement);
    REQUIRE(result->RowCount() == 1);
    REQUIRE(result->Collection().GetValue(0, 0).ToString() == value);
    query(con, "SET enable_duckdb_fallback = false");
    REQUIRE(con.Query(statement)->HasError());
  }
}

TEST_CASE("DuckLake Parquet scans honor the configured batch target",
          "[isolated_context][ducklake][ducklake_sql]")
{
  if (!sirius::ducklake::bridge_available()) SKIP("DuckLake private binding bridge is disabled");
  auto target = GENERATE(uint64_t{1}, uint64_t{64 * 1024 * 1024});
  INFO(target);
  sirius::test::scoped_temp_directory disk;
  auto path = write_lake_config(disk.directory, [&](YAML::Node& config) {
    config["sirius"]["operator_params"]["scan_task_batch_size"] = target;
  });
  sirius::test::shared_test_env env(path);
  auto con = env.make_connection();
  if (!load_ducklake(con)) return;
  attach_lake(con, disk.directory);
  query(con, "SET gpu_execution=false");
  query(con, "CALL lake.set_option('data_inlining_row_limit', 0)");
  query(con, "CREATE TABLE lake.q AS SELECT i::BIGINT id FROM range(524288) r(i)");
  auto sql              = std::string("SELECT SUM(id) FROM lake.q");
  auto expected         = ducklake_result_bag(*query(con, sql));
  auto runtime          = sirius::test::get_registered_sirius_context(con);
  auto counters         = runtime->physical_counters();
  counters->track_units = true;
  std::atomic<unsigned> decoded{0};
  counters->parquet_phase_for_testing = [&](std::string const&, bool footer) {
    if (!footer) ++decoded;
  };
  auto clear =
    std::shared_ptr<void>(nullptr, [&](void*) { counters->parquet_phase_for_testing = {}; });
  query(con, "SET gpu_execution=true");
  query(con, "SET enable_duckdb_fallback=false");
  for (auto const& statement :
       {sql, "CALL gpu_execution(" + duckdb::Value(sql).ToSQLString() + ")"}) {
    decoded = 0;
    CHECK(ducklake_result_bag(*query(con, statement)) == expected);
    if (target == 1) {
      CHECK(decoded.load() > 1);
    } else {
      CHECK(decoded.load() == 1);
    }
  }
}

TEST_CASE_METHOD(ducklake_fixture,
                 "Production DuckLake SQL lowers complete bound scans",
                 "[isolated_context][ducklake][ducklake_sql]")
{
  auto shape = GENERATE(std::string("parquet"), std::string("mixed"), std::string("inline"));
  INFO(shape);
  if (!sirius::ducklake::bridge_available()) SKIP("DuckLake private binding bridge is disabled");
  attach_lake(con, disk.directory);
  run("SET gpu_execution=false");
  run("CALL lake.set_option('data_inlining_row_limit', 100)");
  run("CREATE TABLE lake.q(id BIGINT, value BIGINT, label VARCHAR)");
  run("INSERT INTO lake.q VALUES (1,11,'one'),(2,22,'two'),(2,22,'two'),(3,NULL,'three')");
  if (shape != "inline") run("CALL ducklake_flush_inlined_data('lake')");
  run("INSERT INTO lake.q VALUES (4,44,'four'),(5,55,NULL)");
  if (shape == "parquet") run("CALL ducklake_flush_inlined_data('lake')");
  run("ATTACH '" + (std::filesystem::path(disk.directory) / "native.db").string() +
      "' AS native_store");
  run("CREATE TABLE native_store.native_keys(id BIGINT)");
  run("INSERT INTO native_store.native_keys VALUES (2),(4)");
  run("CHECKPOINT native_store");
  run("SET gpu_execution=true");
  auto runtime          = sirius::test::get_registered_sirius_context(con);
  auto counters         = runtime->physical_counters();
  counters->track_units = true;
  // Admission must not depend on diagnostic counters being enabled.
  run("SET enable_duckdb_fallback=false");
  for (bool track : {false, true}) {
    counters->track_units = track;
    auto before           = runtime->get_transparent_execution_stats();
    CHECK(run("SELECT id FROM lake.q")->RowCount() == 6);
    CHECK(run("CALL gpu_execution('SELECT id FROM lake.q')")->RowCount() == 6);
    auto after = runtime->get_transparent_execution_stats();
    CHECK(after.executions == before.executions + 1);
    CHECK(after.fallbacks == before.fallbacks);
    CHECK(after.runtime_fallbacks == before.runtime_fallbacks);
  }
  std::atomic<unsigned> decoded{0};
  counters->parquet_phase_for_testing = [&](std::string const&, bool footer) {
    if (!footer) ++decoded;
  };
  auto clear =
    std::shared_ptr<void>(nullptr, [&](void*) { counters->parquet_phase_for_testing = {}; });
  run("SET enable_duckdb_fallback=false");
  std::vector<std::pair<std::string, bool>> queries = {
    {"SELECT id FROM lake.q", true},
    {"SELECT 1 FROM lake.q", true},
    {"SELECT value,id,label FROM lake.q WHERE id>=2", false},
    {"SELECT id FROM lake.q WHERE id=1 OR id=4", false},
    {"SELECT id,value FROM lake.q ORDER BY id DESC, value ASC NULLS LAST LIMIT 3", false},
    {"SELECT COUNT(*) FROM lake.q", false},
    {"SELECT COUNT(*) FROM lake.q WHERE id>=2", false},
    {"SELECT COUNT(*) FROM lake.q WHERE id%2=0", false},
    {"SELECT COUNT(*) FROM lake.q WHERE id>90", false},
    {"SELECT id,COUNT(*) FROM lake.q GROUP BY id", true},
    {"SELECT a.id,b.value FROM lake.q a JOIN lake.q b ON a.id=b.id WHERE b.id>=2", true},
    {"SELECT a.id FROM lake.q a JOIN native_store.native_keys b ON a.id=b.id", false},
    {"SELECT id FROM lake.q WHERE id>90", false}};
  for (auto const& [sql, both_entries] : queries) {
    INFO(sql);
    auto expected = cpu_result(sql);
    for (bool explicit_entry : {false, true}) {
      if (explicit_entry && !both_entries) continue;
      INFO(explicit_entry);
      auto decoded_before  = decoded.load();
      auto imported_before = counters->arrow_rows_imported.load();
      check_gpu_result(sql, expected, explicit_entry);
      if (sql == "SELECT id FROM lake.q" || sql == "SELECT 1 FROM lake.q") {
        CHECK((decoded.load() > decoded_before) == (shape != "inline"));
        CHECK((counters->arrow_rows_imported.load() > imported_before) == (shape != "parquet"));
      }
    }
  }
  auto prepared = con.Prepare("SELECT id FROM lake.q WHERE id>=?");
  REQUIRE_FALSE(prepared->HasError());
  for (int64_t value : {2, 4, 1}) {
    auto expected = cpu_result("SELECT id FROM lake.q WHERE id>=" + std::to_string(value));
    check_gpu_execution(expected, [&] { return prepared->Execute(value); });
  }
}
TEST_CASE_METHOD(ducklake_fixture,
                 "Production DuckLake SQL retires failed GPU work before replay",
                 "[isolated_context][ducklake][ducklake_recovery]")
{
  auto explicit_entry = GENERATE(false, true);
  INFO(explicit_entry);
  auto mode = GENERATE(std::string("replay"),
                       std::string("disabled"),
                       std::string("replay_failure"),
                       std::string("interrupted"));
  INFO(mode);
  if (!sirius::ducklake::bridge_available()) SKIP("DuckLake private binding bridge is disabled");
  create_recovery_table(4);
  std::string sql = "SELECT id,value FROM lake.q WHERE id%3<>0";
  auto expected   = ducklake_result_bag(*run(sql));
  run("SET gpu_execution=true");
  auto runtime          = sirius::test::get_registered_sirius_context(con);
  auto counters         = runtime->physical_counters();
  counters->track_units = true;
  auto hosts =
    runtime->get_memory_manager().get_memory_spaces_for_tier(cucascade::memory::Tier::HOST);
  REQUIRE(hosts.size() == 1);
  auto reserved = hosts[0]->get_total_reserved_memory();
  std::atomic<unsigned> decoded{0}, replay_checks{0};
  uint64_t reserved_at_replay = UINT64_MAX;
  std::string failed_file;
  counters->parquet_phase_for_testing = [&](std::string const& file, bool footer) {
    if (!footer && ++decoded == 3) {
      failed_file = file;
      throw sirius::transparent::classified_execution_error(
        sirius::transparent::late_failure_cause::reader_io, "injected DuckLake decode failure");
    }
  };
  runtime->cpu_replay_hook_for_testing = [&] {
    ++replay_checks;
    reserved_at_replay = hosts[0]->get_total_reserved_memory();
    if (mode == "replay_failure") std::filesystem::rename(failed_file, failed_file + ".saved");
  };
  auto clear = std::shared_ptr<void>(nullptr, [&](void*) {
    counters->parquet_phase_for_testing  = {};
    runtime->cpu_replay_hook_for_testing = {};
    if (!failed_file.empty() && std::filesystem::exists(failed_file + ".saved"))
      std::filesystem::rename(failed_file + ".saved", failed_file);
  });
  run("SET sirius_test_sync_cpu_replay=true");
  run(std::string("SET sirius_test_interrupt_before_replay=") +
      (mode == "interrupted" ? "true" : "false"));
  run(std::string("SET enable_duckdb_fallback=") + (mode == "disabled" ? "false" : "true"));
  auto before = runtime->get_transparent_execution_stats();
  auto result = con.Query(
    explicit_entry ? "CALL gpu_execution(" + duckdb::Value(sql).ToSQLString() + ")" : sql);
  INFO((result->HasError() ? result->GetError() : "replayed successfully"));
  CHECK(decoded >= 3);
  if (mode == "replay") {
    REQUIRE_FALSE(result->HasError());
    CHECK(ducklake_result_bag(*result) == expected);
  } else
    CHECK(result->HasError());
  bool replayed = mode == "replay" || mode == "replay_failure";
  CHECK(replay_checks == (mode == "disabled" ? 0 : 1));
  if (mode != "disabled") CHECK(reserved_at_replay == reserved);
  result.reset();
  CHECK(hosts[0]->get_total_reserved_memory() == reserved);
  auto after = runtime->get_transparent_execution_stats();
  if (mode == "interrupted") {
    auto cancelled = static_cast<size_t>(sirius::transparent::late_failure_condition::cancelled);
    CHECK(after.late_failure_no_replay[cancelled] == before.late_failure_no_replay[cancelled] + 1);
  }
  CHECK(after.runtime_fallbacks == before.runtime_fallbacks + (replayed ? 1 : 0));
  counters->parquet_phase_for_testing  = {};
  runtime->cpu_replay_hook_for_testing = {};
  if (mode == "replay_failure") std::filesystem::rename(failed_file + ".saved", failed_file);
  run("SET sirius_test_sync_cpu_replay=false");
  run("SET sirius_test_interrupt_before_replay=false");
  run("SET enable_duckdb_fallback=false");
  auto next = runtime->get_transparent_execution_stats();
  CHECK(ducklake_result_bag(*run(sql)) == expected);
  CHECK(runtime->get_transparent_execution_stats().executions == next.executions + 1);
}
TEST_CASE_METHOD(ducklake_fixture,
                 "Production DuckLake SQL preserves its view across concurrent commits",
                 "[isolated_context][ducklake][ducklake_concurrency]")
{
  auto mode = GENERATE(std::string("commit"), std::string("flush"));
  INFO(mode);
  if (!sirius::ducklake::bridge_available()) SKIP("DuckLake private binding bridge is disabled");
  attach_lake(con, disk.directory);
  run("SET gpu_execution=false");
  run("CALL lake.set_option('data_inlining_row_limit', 100)");

  run("CREATE TABLE lake.q(id BIGINT)");
  run("INSERT INTO lake.q VALUES (1),(2),(2)");
  run("CALL ducklake_flush_inlined_data('lake')");
  run("INSERT INTO lake.q VALUES (3),(4)");
  std::string sql = "SELECT id FROM lake.q";
  auto expected   = ducklake_result_bag(*run(sql));
  run("SET gpu_execution=true");
  auto runtime          = sirius::test::get_registered_sirius_context(con);
  auto counters         = runtime->physical_counters();
  counters->track_units = true;
  auto writer           = env.make_connection();
  auto write            = [&](std::string const& text) {
    auto r = writer.Query(text);
    if (r->HasError()) throw std::runtime_error(r->GetError());
  };
  write("SET gpu_execution=false");
  run("SET enable_duckdb_fallback=false");
  query_pause pause;
  std::atomic<bool> first{true};
  counters->after_certify_for_testing = [&] {
    if (first.exchange(false)) { pause.arrive_and_wait(); }
  };
  auto clear =
    std::shared_ptr<void>(nullptr, [&](void*) { counters->after_certify_for_testing = {}; });
  auto before = runtime->get_transparent_execution_stats();
  auto active =
    std::async(std::launch::async, [&] { return ducklake_result_bag(*con.Query(sql)); });
  auto unblock = std::shared_ptr<void>(nullptr, [&](void*) { pause.release(); });
  REQUIRE(pause.wait_until_arrived());
  write("INSERT INTO lake.q VALUES (9)");
  if (mode == "flush") write("CALL ducklake_flush_inlined_data('lake')");
  pause.release();
  REQUIRE(active.wait_for(std::chrono::seconds(15)) == std::future_status::ready);
  CHECK(active.get() == expected);
  counters->after_certify_for_testing = {};
  auto after                          = runtime->get_transparent_execution_stats();
  CHECK(after.executions == before.executions + 1);
  CHECK(after.runtime_fallbacks == before.runtime_fallbacks);
  expected.push_back("9|");
  std::sort(expected.begin(), expected.end());
  CHECK(ducklake_result_bag(*run(sql)) == expected);
}

TEST_CASE_METHOD(ducklake_fixture,
                 "Production DuckLake SQL cancellation drains blocked reads without replay",
                 "[isolated_context][ducklake][ducklake_cancel]")
{
  auto footer = GENERATE(false, true);
  INFO(footer);
  if (!sirius::ducklake::bridge_available()) SKIP("DuckLake private binding bridge is disabled");
  attach_lake(con, disk.directory);
  run("SET gpu_execution=false");
  run("CALL lake.set_option('data_inlining_row_limit', 100)");

  run("CREATE TABLE lake.q AS SELECT i::BIGINT id FROM range(10000) r(i)");
  run("CALL ducklake_flush_inlined_data('lake')");
  run("INSERT INTO lake.q VALUES (10001)");
  run("SET gpu_execution=true");
  auto runtime          = sirius::test::get_registered_sirius_context(con);
  auto counters         = runtime->physical_counters();
  counters->track_units = true;
  auto hosts =
    runtime->get_memory_manager().get_memory_spaces_for_tier(cucascade::memory::Tier::HOST);
  REQUIRE(hosts.size() == 1);
  auto reserved = hosts[0]->get_total_reserved_memory();
  query_pause pause;
  std::atomic<bool> first{true};
  std::atomic<unsigned> replays{0};
  counters->parquet_phase_for_testing = [&](std::string const&, bool at_footer) {
    if (at_footer == footer && first.exchange(false)) { pause.arrive_and_wait(); }
  };
  runtime->cpu_replay_hook_for_testing = [&] { ++replays; };
  auto clear                           = std::shared_ptr<void>(nullptr, [&](void*) {
    counters->parquet_phase_for_testing  = {};
    runtime->cpu_replay_hook_for_testing = {};
  });
  run("SET sirius_test_sync_cpu_replay=true");
  auto before = runtime->get_transparent_execution_stats();
  auto active =
    std::async(std::launch::async, [&] { return con.Query("SELECT id FROM lake.q WHERE id%2=0"); });
  auto unblock = std::shared_ptr<void>(nullptr, [&](void*) { pause.release(); });
  REQUIRE(pause.wait_until_arrived());
  con.Interrupt();
  pause.release();
  REQUIRE(active.wait_for(std::chrono::seconds(15)) == std::future_status::ready);
  auto result = active.get();
  REQUIRE(result->HasError());
  INFO(result->GetError());
  CHECK(result->GetError().find("Interrupt") != std::string::npos);
  CHECK(replays == 0);
  CHECK(runtime->get_transparent_execution_stats().runtime_fallbacks == before.runtime_fallbacks);
  result.reset();
  CHECK(hosts[0]->get_total_reserved_memory() == reserved);
  counters->parquet_phase_for_testing = {};
  run("SET sirius_test_sync_cpu_replay=false");
  run("SET enable_duckdb_fallback=false");
  CHECK(ducklake_result_bag(*run("SELECT id FROM lake.q WHERE id=10001")) ==
        std::vector<std::string>{"10001|"});
}
TEST_CASE_METHOD(ducklake_fixture,
                 "Production DuckLake SQL combines dynamic filters with local visibility",
                 "[isolated_context][ducklake][ducklake_dynamic]")
{
  auto [shape, deletes] = GENERATE(table<std::string, std::string>({{"parquet", "ordinary"},
                                                                    {"parquet", "dv"},
                                                                    {"parquet", "inline"},
                                                                    {"mixed", "ordinary"},
                                                                    {"mixed", "dv"},
                                                                    {"mixed", "inline"},
                                                                    {"inline", "inline"}}));
  INFO(shape);
  INFO(deletes);
  if (!sirius::ducklake::bridge_available()) SKIP("DuckLake private binding bridge is disabled");
  attach_lake(con, disk.directory);
  run("SET gpu_execution=false");
  run("CALL lake.set_option('data_inlining_row_limit'," +
      std::string(shape == "inline" ? "20000" : "0") + ")");
  run("CALL lake.set_option('parquet_row_group_size',2048)");
  run("CALL lake.set_option('write_deletion_vectors'," +
      std::string(deletes == "dv" ? "true" : "false") + ")");
  run(
    "CREATE TABLE lake.q AS SELECT range::BIGINT id, CASE WHEN range%11=0 THEN NULL ELSE range%17 "
    "END::BIGINT value FROM range(10000)");
  if (shape == "mixed") {
    run("CALL lake.set_option('data_inlining_row_limit',20000)");
    run("INSERT INTO lake.q SELECT range::BIGINT, (range%17)::BIGINT FROM range(2700,3300)");
  }
  run("CALL lake.set_option('data_inlining_row_limit'," +
      std::string(deletes == "inline" ? "20000" : "0") + ")");
  run("DELETE FROM lake.q WHERE id%7=0");
  auto delete_files = run(
    "SELECT format FROM __ducklake_metadata_lake.ducklake_delete_file "
    "WHERE end_snapshot IS NULL");
  REQUIRE(delete_files->RowCount() == (deletes == "inline" ? 0 : 1));
  if (deletes != "inline")
    CHECK(delete_files->Collection().GetValue(0, 0).ToString() ==
          (deletes == "dv" ? "puffin" : "parquet"));
  run("ATTACH '" + (std::filesystem::path(disk.directory) / "keys.db").string() +
      "' AS native_store");
  run("CREATE TABLE native_store.keys AS SELECT range::BIGINT k FROM range(17)");
  run("CHECKPOINT native_store");
  std::string sql =
    "SELECT q.value,q.id FROM lake.q q JOIN native_store.keys b ON q.value=b.k WHERE q.id>=2048 "
    "AND q.id<4096 AND b.k%4=1";
  auto expected = ducklake_result_bag(*run(sql));
  run("SET gpu_execution=true");
  auto runtime          = sirius::test::get_registered_sirius_context(con);
  auto counters         = runtime->physical_counters();
  counters->track_units = true;
  std::atomic<unsigned> decoded{0};
  counters->parquet_phase_for_testing = [&](std::string const&, bool footer) {
    if (!footer) ++decoded;
  };
  auto clear =
    std::shared_ptr<void>(nullptr, [&](void*) { counters->parquet_phase_for_testing = {}; });
  run("SET enable_duckdb_fallback=false");
  run("SET enable_dynamic_filter=true");
  run("SET dynamic_filter_domain_coverage_threshold=2.0");
  for (int repeat = 0; repeat < 2; ++repeat) {
    auto dynamic_before = runtime->get_dynamic_filter_stats_snapshot();
    check_gpu_result(sql, expected);
    auto dynamic_after = runtime->get_dynamic_filter_stats_snapshot();
    CHECK(dynamic_after.producers_enabled > dynamic_before.producers_enabled);
    CHECK(dynamic_after.filters_pushed > dynamic_before.filters_pushed);
    CHECK(dynamic_after.publications_failed == dynamic_before.publications_failed);
  }
  CHECK((decoded.load() > 0) == (shape != "inline"));
  CHECK((counters->arrow_rows_imported.load() > 0) == (shape != "parquet"));
}

TEST_CASE_METHOD(ducklake_fixture,
                 "Production DuckLake SQL refuses a grant that cannot progress both join scans",
                 "[isolated_context][ducklake][ducklake_admission]")
{
  auto fallback = GENERATE(false, true);
  INFO(fallback);
  if (!sirius::ducklake::bridge_available()) SKIP("DuckLake private binding bridge is disabled");
  attach_lake(con, disk.directory);
  run("SET gpu_execution=false");
  run("CALL lake.set_option('data_inlining_row_limit', 100)");

  run("CREATE TABLE lake.q(id BIGINT)");
  run("INSERT INTO lake.q VALUES (1),(2),(2),(3)");
  std::string sql = "SELECT a.id,b.id FROM lake.q a JOIN lake.q b ON a.id=b.id";
  auto expected   = ducklake_result_bag(*run(sql));
  run("SET gpu_execution=true");
  auto runtime          = sirius::test::get_registered_sirius_context(con);
  auto counters         = runtime->physical_counters();
  counters->track_units = true;
  auto provider         = std::make_shared<sirius::scan_manager::test::test_reservation_provider>();
  auto envelope         = sirius::ducklake::scan_ingestible::envelope(1, 0);
  provider->grant_bytes = 2 * envelope.retained;
  for (auto bytes : envelope.temporary)
    provider->grant_bytes += bytes;
  counters->preparation_provider_for_testing = provider;
  auto clear                                 = std::shared_ptr<void>(
    nullptr, [&](void*) { counters->preparation_provider_for_testing.reset(); });
  run(std::string("SET enable_duckdb_fallback=") + (fallback ? "true" : "false"));
  auto result = con.Query(sql);
  if (fallback) {
    REQUIRE_FALSE(result->HasError());
    CHECK(ducklake_result_bag(*result) == expected);
  } else
    CHECK(result->HasError());
  CHECK(provider->seen->requests == 1);
  CHECK(provider->seen->allocations == 0);
  CHECK(provider->outstanding() == 0);
  CHECK(counters->arrow_batches_imported == 0);
}

TEST_CASE("DuckLake joins refuse an insufficient preparation permit ceiling before data work",
          "[isolated_context][ducklake][ducklake_admission_minimum]")
{
  if (!sirius::ducklake::bridge_available()) SKIP("DuckLake private binding bridge is disabled");
  sirius::test::scoped_temp_directory disk;
  auto config_path = write_lake_config(disk.directory, [](YAML::Node& config) {
    config["sirius"]["executor"]["scan_manager"]["preparation"]["max_inflight_dv_jobs"] = 1;
  });
  sirius::test::shared_test_env env(config_path);
  auto con = env.make_connection();
  if (!load_ducklake(con)) return;
  query(con, "SET gpu_execution=false");
  attach_lake(con, disk.directory);
  query(con, "CREATE TABLE lake.q AS SELECT 42::BIGINT id");
  auto runtime          = sirius::test::get_registered_sirius_context(con);
  auto counters         = runtime->physical_counters();
  counters->track_units = true;
  bool refused          = false;
  counters->iceberg_statement_route_for_testing =
    [&](std::string_view route, sirius::scan_manager::admission_decision const& decision) {
      CHECK(route == "provider_admission_refused");
      refused =
        !decision.deferred && decision.reason == sirius::scan_manager::admission_reason::capacity;
    };
  auto clear = std::shared_ptr<void>(
    nullptr, [&](void*) { counters->iceberg_statement_route_for_testing = {}; });
  auto imported = counters->arrow_batches_imported.load();
  query(con, "SET gpu_execution=true");
  query(con, "SET enable_duckdb_fallback=false");
  auto result = con.Query("SELECT count(*) FROM lake.q a JOIN lake.q b ON a.id=b.id");
  REQUIRE(result->HasError());
  CHECK(refused);
  CHECK(counters->arrow_batches_imported.load() == imported);
  CHECK(runtime->get_scan_manager().num_active_queries() == 0);
}

TEST_CASE("Production DuckLake SQL isolates concurrent prepared executions",
          "[isolated_context][ducklake][ducklake_concurrency]")
{
  if (!sirius::ducklake::bridge_available()) SKIP("DuckLake private binding bridge is disabled");
  sirius::test::scoped_temp_directory disk;
  auto config_path = write_lake_config(disk.directory, [](YAML::Node& config) {
    config["sirius"]["max_concurrent_queries"] = 2;
    if (config["sirius"]["executor"]["scan_manager"]["max_concurrent_queries"])
      config["sirius"]["executor"]["scan_manager"]["max_concurrent_queries"] = 2;
  });
  sirius::test::shared_test_env env(config_path);
  auto con = env.make_connection();
  if (!load_ducklake(con)) return;
  auto run = [&](std::string const& sql) { return query(con, sql); };
  attach_lake(con, disk.directory);
  run("SET gpu_execution=false");
  run("CALL lake.set_option('data_inlining_row_limit', 100)");

  run("CREATE TABLE lake.q AS SELECT range::BIGINT id,(range%17)::BIGINT value FROM range(2048)");
  run("CALL ducklake_flush_inlined_data('lake')");
  run("INSERT INTO lake.q VALUES (7,70),(8,80)");
  std::array<int64_t, 4> limits{0, 500, 1000, 1500};
  std::vector<std::vector<std::string>> expected;
  for (auto v : limits)
    expected.push_back(
      ducklake_result_bag(*run("SELECT value,id FROM lake.q WHERE id>=" + std::to_string(v) +
                               " AND id<" + std::to_string(v + 300))));
  run("SET gpu_execution=true");
  auto runtime  = sirius::test::get_registered_sirius_context(con);
  auto observed = std::make_shared<observed_host_reservations>(runtime->get_memory_manager());
  runtime->physical_counters()->track_units                      = true;
  runtime->physical_counters()->preparation_provider_for_testing = observed;
  runtime->physical_counters()->iceberg_statement_route_for_testing =
    [&](std::string_view, sirius::scan_manager::admission_decision const& decision) {
      if (!decision.deferred) return;
      std::lock_guard lock(observed->seen->mutex);
      observed->seen->exceeded |=
        decision.sigma_retained + decision.n_permits * decision.w > decision.c_obtained ||
        decision.n_permits < 1;
    };
  auto clear_provider   = std::shared_ptr<void>(nullptr, [&](void*) {
    runtime->physical_counters()->preparation_provider_for_testing.reset();
    runtime->physical_counters()->iceberg_statement_route_for_testing = {};
  });
  auto counters         = runtime->physical_counters();
  counters->track_units = true;
  auto clear =
    std::shared_ptr<void>(nullptr, [&](void*) { counters->parquet_phase_for_testing = {}; });
  auto sibling = env.make_connection();
  run("SET enable_duckdb_fallback=false");
  auto setting = sibling.Query("SET enable_duckdb_fallback=false");
  REQUIRE_FALSE(setting->HasError());
  REQUIRE(runtime->get_config().max_concurrent_queries() == 2);
  std::mutex rendezvous_mutex;
  std::condition_variable rendezvous;
  unsigned arrived = 0;
  std::atomic<unsigned> first_footers{0};
  counters->parquet_phase_for_testing = [&](std::string const&, bool footer) {
    if (!footer || first_footers.fetch_add(1) >= 2) return;
    std::unique_lock lock(rendezvous_mutex);
    ++arrived;
    rendezvous.notify_all();
    if (!rendezvous.wait_for(lock, std::chrono::seconds(15), [&] { return arrived == 2; }))
      throw std::runtime_error("two DuckLake queries did not reach preparation concurrently");
  };
  auto before = runtime->get_transparent_execution_stats();
  std::promise<void> release;
  auto ready   = release.get_future().share();
  auto execute = [&](duckdb::Connection& con, int shift) {
    ready.wait();
    auto prepared = con.Prepare("SELECT value,id FROM lake.q WHERE id>=? AND id<?");
    if (prepared->HasError()) throw std::runtime_error(prepared->GetError());
    for (int repeat = 0; repeat < 8; ++repeat) {
      auto index  = (repeat + shift) % limits.size();
      auto v      = limits[index];
      auto result = prepared->Execute(v, v + 300);
      if (ducklake_result_bag(*result) != expected[index]) return false;
    }
    return true;
  };
  auto a = std::async(std::launch::async, [&] { return execute(con, 0); });
  auto b = std::async(std::launch::async, [&] { return execute(sibling, 2); });
  release.set_value();
  REQUIRE(a.wait_for(std::chrono::seconds(30)) == std::future_status::ready);
  REQUIRE(b.wait_for(std::chrono::seconds(30)) == std::future_status::ready);
  CHECK(a.get());
  CHECK(b.get());
  auto after = runtime->get_transparent_execution_stats();
  CHECK(after.executions == before.executions + 16);
  CHECK(after.fallbacks == before.fallbacks);
  CHECK(after.runtime_fallbacks == before.runtime_fallbacks);
  CHECK(arrived == 2);
  CHECK_FALSE(observed->seen->exceeded);
  CHECK(observed->seen->peak_live > 0);
  CHECK(observed->seen->live == 0);
  CHECK(observed->seen->grants == 0);
  CHECK(observed->seen->peak_active >= 2);
}

TEST_CASE("Concurrent DuckLake scans release input budgets while GPU outputs remain retained",
          "[isolated_context][ducklake][ducklake_budget_retention]")
{
  if (!sirius::ducklake::bridge_available()) SKIP("DuckLake private binding bridge is disabled");
  auto deletion = GENERATE(std::string("inline"), std::string("parquet"), std::string("puffin"));
  auto join     = GENERATE(false, true);
  CAPTURE(deletion, join);
  sirius::test::scoped_temp_directory disk;
  auto config_path = write_lake_config(disk.directory, [](YAML::Node& config) {
    config["sirius"]["max_concurrent_queries"]                                          = 2;
    config["sirius"]["executor"]["scan_manager"]["max_concurrent_queries"]              = 2;
    config["sirius"]["executor"]["scan_manager"]["preparation"]["max_inflight_dv_jobs"] = 2;
  });
  sirius::test::shared_test_env env(config_path);
  auto con = env.make_connection();
  if (!load_ducklake(con)) return;
  query(con, "SET gpu_execution=false");
  attach_lake(con, disk.directory);
  query(con, "CALL lake.set_option('data_inlining_row_limit',100)");
  query(con, "CREATE TABLE lake.q(id BIGINT)");
  for (int file = 0; file < 4; ++file)
    query(con,
          "INSERT INTO lake.q SELECT i FROM range(" + std::to_string(file * 4096) + "," +
            std::to_string((file + 1) * 4096) + ") r(i)");
  query(con,
        "CALL lake.set_option('write_deletion_vectors'," +
          std::string(deletion == "puffin" ? "true" : "false") + ")");
  query(con,
        "CALL lake.set_option('data_inlining_row_limit'," +
          std::string(deletion == "inline" ? "20000" : "0") + ")");
  query(con, "DELETE FROM lake.q WHERE id%7=0");
  query(con, "CALL lake.set_option('data_inlining_row_limit',100)");
  query(con, "INSERT INTO lake.q VALUES (-1),(-2),(NULL)");
  auto const sql = std::string(join ? "SELECT a.id FROM lake.q a JOIN lake.q b ON a.id=b.id"
                                    : "SELECT id FROM lake.q");
  auto expected  = ducklake_result_bag(*query(con, sql));
  auto sibling   = env.make_connection();
  for (auto* connection : {&con, &sibling}) {
    query(*connection, "SET gpu_execution=false");
    query(*connection, "SET enable_duckdb_fallback=false");
    connection->context->registered_state->Insert(
      sirius::exec::stream_bind_catalog::kStateKey,
      duckdb::make_shared_ptr<sirius::exec::stream_bind_catalog>());
  }
  auto runtime          = sirius::test::get_registered_sirius_context(con);
  auto counters         = runtime->physical_counters();
  counters->track_units = true;
  auto observed = std::make_shared<observed_host_reservations>(runtime->get_memory_manager());
  counters->preparation_provider_for_testing = observed;
  std::atomic<unsigned> admissions{0}, decoded{0}, first_footers{0}, visits{0}, gate_checks{0};
  std::atomic<size_t> retained_batches{0};
  counters->iceberg_statement_route_for_testing =
    [&](std::string_view, sirius::scan_manager::admission_decision const& decision) {
      if (!decision.deferred) throw std::runtime_error("mixed scan admission was refused");
      ++admissions;
      observed->record_admission(decision, join ? 2 : 1, 2);
    };
  std::mutex mutex;
  std::condition_variable cv;
  unsigned footers = 0, round_arrivals = 0, round = 0, probes_arrived = 0;
  counters->parquet_phase_for_testing = [&](std::string const&, bool footer) {
    if (!footer) {
      ++decoded;
      return;
    }
    if (first_footers.fetch_add(1) >= 2) return;
    std::unique_lock lock(mutex);
    ++footers;
    cv.notify_all();
    if (!cv.wait_for(lock, std::chrono::seconds(15), [&] { return footers == 2; }))
      throw std::runtime_error("mixed scans did not prepare concurrently");
  };
  auto observer          = duckdb::make_shared_ptr<provider_query_observer>();
  auto clear             = std::shared_ptr<void>(nullptr, [&](void*) {
    observer->inspect_plan = {};
    counters->preparation_provider_for_testing.reset();
    counters->iceberg_statement_route_for_testing = {};
    counters->parquet_phase_for_testing           = {};
  });
  observer->inspect_plan = [&](duckdb::ClientContext& context, duckdb::LogicalOperator& plan) {
    // Provider SQL may reuse the statement context; only the outer binding starts a fragment.
    if (&context != con.context.get() && &context != sibling.context.get()) return;
    static thread_local bool inside = false;
    if (inside) return;
    inside     = true;
    auto leave = std::shared_ptr<void>(nullptr, [&](void*) { inside = false; });
    ++visits;
    auto inspect = [&](auto&& self, duckdb::LogicalOperator& node) -> void {
      if (node.type == duckdb::LogicalOperatorType::LOGICAL_GET) {
        auto& get = node.Cast<duckdb::LogicalGet>();
        if (get.function.GetName() == "ducklake_scan") {
          check_mixed_input_gate(
            sirius::ducklake::binding_handle::capture(get.function, get.bind_data.get(), context),
            *observed,
            deletion);
          ++gate_checks;
        }
      }
      for (auto& child : node.children)
        self(self, *child);
    };
    inspect(inspect, plan);
    {
      std::unique_lock lock(mutex);
      if (++probes_arrived == 2) {
        std::lock_guard charges(observed->seen->mutex);
        observed->seen->exceeded |= observed->seen->live != 0 || observed->seen->grants != 0;
        observed->seen->peak_active = 0;
        observed->seen->peak_live   = 0;
        observed->seen->peak_grants = 0;
        cv.notify_all();
      } else if (!cv.wait_for(
                   lock, std::chrono::seconds(15), [&] { return probes_arrived == 2; })) {
        throw std::runtime_error("mixed gate probes did not finish");
      }
    }
    // Keep the fragment resources and all prior GPU outputs alive through later scans.
    std::vector<std::unique_ptr<sirius::exec::streaming_fragment>> retained;
    for (unsigned repeat = 0; repeat < 3; ++repeat) {
      sirius::exec::fragment_spec spec;
      spec.outputs     = {0};
      spec.plan_source = [&](duckdb::ClientContext& current) {
        return sirius::exec::bound_plan{plan.Copy(current), nullptr};
      };
      auto fragment = std::make_unique<sirius::exec::streaming_fragment>(context, std::move(spec));
      fragment->build();
      fragment->run();
      if (fragment->output_row_count(0) != expected.size() || fragment->output_batch_count(0) == 0)
        throw std::runtime_error("mixed fragment did not retain the expected GPU output");
      retained_batches += fragment->output_batch_count(0);
      retained.push_back(std::move(fragment));
      std::unique_lock lock(mutex);
      if (++round_arrivals == 2) {
        // Both queries have completed, but neither has consumed any GPU output.
        std::lock_guard charges(observed->seen->mutex);
        observed->seen->exceeded |= observed->seen->live != 0 || observed->seen->grants != 0;
        round_arrivals = 0;
        ++round;
        cv.notify_all();
      } else if (!cv.wait_for(lock, std::chrono::seconds(30), [&] { return round > repeat; })) {
        throw std::runtime_error("mixed scan stalled while earlier GPU outputs were retained");
      }
    }
    // Validate values after later scans reused the input capacity, not just row counts.
    for (auto& fragment : retained) {
      std::vector<std::string> actual;
      while (auto batch = fragment->pull(0)) {
        auto column = sirius::get_cudf_table_view(**batch).column(0);
        std::vector<int64_t> values(column.size());
        std::vector<cudf::bitmask_type> mask((column.offset() + column.size() + 31) / 32);
        if (cudaMemcpy(values.data(),
                       column.data<int64_t>(),
                       values.size() * sizeof(int64_t),
                       cudaMemcpyDeviceToHost) != cudaSuccess)
          throw std::runtime_error("retained output copy failed");
        if (column.nullable() && cudaMemcpy(mask.data(),
                                            column.null_mask(),
                                            mask.size() * sizeof(cudf::bitmask_type),
                                            cudaMemcpyDeviceToHost) != cudaSuccess)
          throw std::runtime_error("retained output validity copy failed");
        for (size_t i = 0; i < values.size(); ++i) {
          auto bit   = column.offset() + i;
          auto valid = !column.nullable() || (mask[bit / 32] & (uint32_t{1} << (bit % 32)));
          actual.push_back(valid ? std::to_string(values[i]) + "|" : "NULL|");
        }
      }
      std::sort(actual.begin(), actual.end());
      if (actual != expected || !fragment->drained(0))
        throw std::runtime_error("retained GPU values differ from native DuckLake");
    }
  };
  duckdb::OptimizerExtension extension;
  extension.optimizer_info    = observer;
  extension.optimize_function = observe_provider_query;
  duckdb::OptimizerExtension::Register(duckdb::DBConfig::GetConfig(*con.context), extension);
  auto imported = counters->arrow_rows_imported.load();
  auto before   = runtime->get_transparent_execution_stats();
  auto execute  = [&](duckdb::Connection& connection) {
    auto result = connection.Query(sql);
    return ducklake_result_bag(*result) == expected;
  };
  auto a = std::async(std::launch::async, [&] { return execute(con); });
  auto b = std::async(std::launch::async, [&] { return execute(sibling); });
  CHECK(a.get());
  CHECK(b.get());
  INFO("peak charged bytes=" << observed->seen->peak_live
                             << ", peak grants=" << observed->seen->peak_grants
                             << ", retained GPU batches=" << retained_batches.load());
  CHECK(retained_batches >= 6);
  CHECK(visits == 2);
  CHECK(admissions == 6);
  CHECK(gate_checks == (join ? 4 : 2));
  CHECK(round == 3);
  CHECK(decoded >= (join ? 48 : 24));
  CHECK(counters->arrow_rows_imported.load() == imported + (join ? 36 : 18));
  CHECK_FALSE(observed->seen->exceeded);
  CHECK(observed->seen->peak_active == 2);
  CHECK(observed->seen->peak_live > 0);
  CHECK(observed->seen->live == 0);
  CHECK(observed->seen->grants == 0);
  auto after = runtime->get_transparent_execution_stats();
  CHECK(after.fallbacks == before.fallbacks);
  CHECK(after.runtime_fallbacks == before.runtime_fallbacks);
}

TEST_CASE_METHOD(
  ducklake_fixture,
  "Production DuckLake SQL preserves transaction local changes and new statement identities",
  "[isolated_context][ducklake][ducklake_transaction]")
{
  if (!sirius::ducklake::bridge_available()) SKIP("DuckLake private binding bridge is disabled");
  attach_lake(con, disk.directory);
  run("SET gpu_execution=false");
  run("CALL lake.set_option('data_inlining_row_limit', 100)");

  run("CREATE TABLE lake.q(id BIGINT,value BIGINT)");
  run("INSERT INTO lake.q VALUES (1,10),(2,20),(3,30)");
  run("CALL ducklake_flush_inlined_data('lake')");
  run("INSERT INTO lake.q VALUES (4,40),(5,50)");
  run("BEGIN");
  run("INSERT INTO lake.q VALUES (6,60),(6,60)");
  run("DELETE FROM lake.q WHERE id IN (1,4)");
  auto expected = ducklake_result_bag(*run("SELECT value,id FROM lake.q"));
  run("SET gpu_execution=true");
  run("SET enable_duckdb_fallback=false");
  check_gpu_result("SELECT value,id FROM lake.q", expected);
  run("ROLLBACK");
  run("SET gpu_execution=false");
  expected = ducklake_result_bag(*run("SELECT value,id FROM lake.q"));
  run("SET gpu_execution=true");
  auto prepared = con.Prepare("SELECT value,id FROM lake.q");
  REQUIRE_FALSE(prepared->HasError());
  check_gpu_execution(expected, [&] { return prepared->Execute(); });
  run("SET gpu_execution=false");
  run("DELETE FROM lake.q WHERE id=2");
  run("INSERT INTO lake.q VALUES (7,70)");
  expected = ducklake_result_bag(*run("SELECT value,id FROM lake.q"));
  run("SET gpu_execution=true");
  check_gpu_execution(expected, [&] { return prepared->Execute(); });
}

TEST_CASE_METHOD(ducklake_fixture,
                 "Production DuckLake SQL drains publication races before deciding replay",
                 "[isolated_context][ducklake][ducklake_publication]")
{
  auto explicit_entry = GENERATE(false, true);
  auto hold_batch     = GENERATE(false, true);
  auto fallback       = GENERATE(false, true);
  CAPTURE(explicit_entry, hold_batch, fallback);
  if (!sirius::ducklake::bridge_available()) SKIP("DuckLake private binding bridge is disabled");
  create_recovery_table(3);
  std::string sql = "SELECT value,COUNT(*),SUM(id) FROM lake.q WHERE id%3<>0 GROUP BY value";
  auto expected   = ducklake_result_bag(*run(sql));
  run("SET gpu_execution=true");
  auto runtime          = sirius::test::get_registered_sirius_context(con);
  auto counters         = runtime->physical_counters();
  counters->track_units = true;
  auto hosts =
    runtime->get_memory_manager().get_memory_spaces_for_tier(cucascade::memory::Tier::HOST);
  REQUIRE(hosts.size() == 1);
  auto reserved = hosts[0]->get_total_reserved_memory();
  std::atomic<unsigned> footers{0}, replays{0};
  query_pause pause;
  counters->parquet_phase_for_testing = [&](std::string const&, bool footer) {
    if (footer && ++footers == 2) {
      pause.arrive_and_wait(std::chrono::seconds(20));
      throw sirius::transparent::classified_execution_error(
        sirius::transparent::late_failure_cause::reader_io,
        "injected later DuckLake footer failure");
    }
  };
  uint64_t reserved_at_replay          = UINT64_MAX;
  runtime->cpu_replay_hook_for_testing = [&] {
    ++replays;
    reserved_at_replay = hosts[0]->get_total_reserved_memory();
  };
  auto clear = std::shared_ptr<void>(nullptr, [&](void*) {
    counters->parquet_phase_for_testing  = {};
    runtime->cpu_replay_hook_for_testing = {};
  });
  run("SET sirius_test_sync_cpu_replay=true");
  run("SET scan_task_batch_size=1");
  // Enable the shared publication observer without holding a real footer.
  run("SET sirius_test_hold_footer_index=" + std::to_string(UINT64_MAX));
  run(std::string("SET sirius_test_hold_published_batch=") + (hold_batch ? "true" : "false"));
  run(std::string("SET enable_duckdb_fallback=") + (fallback ? "true" : "false"));
  auto before    = runtime->get_transparent_execution_stats();
  auto active    = std::async(std::launch::async, [&] {
    return con.Query(explicit_entry ? "CALL gpu_execution(" + duckdb::Value(sql).ToSQLString() + ")"
                                    : sql);
  });
  bool reached   = pause.wait_until_arrived();
  bool published = reached && runtime->get_scan_manager().wait_for_publication_for_testing(
                                std::chrono::seconds(15));
  pause.release();
  auto result = active.get();
  REQUIRE(reached);
  REQUIRE(published);
  INFO((result->HasError() ? result->GetError() : "replayed successfully"));
  CHECK(result->HasError() == !fallback);
  if (fallback) CHECK(ducklake_result_bag(*result) == expected);
  CHECK(replays == (fallback ? 1 : 0));
  if (fallback) CHECK(reserved_at_replay == reserved);
  result.reset();
  CHECK(hosts[0]->get_total_reserved_memory() == reserved);
  auto after = runtime->get_transparent_execution_stats();
  CHECK(after.runtime_fallbacks == before.runtime_fallbacks + (fallback ? 1 : 0));
  if (hold_batch) CHECK(after.discarded_speculative_work > before.discarded_speculative_work);
}

TEST_CASE_METHOD(ducklake_fixture,
                 "Production DuckLake SQL refuses replay when native checkpoint cleanup fails",
                 "[isolated_context][ducklake][ducklake_cleanup]")
{
  auto explicit_entry = GENERATE(false, true);
  CAPTURE(explicit_entry);
  if (!sirius::ducklake::bridge_available()) SKIP("DuckLake private binding bridge is disabled");
  create_recovery_table(4, /*flush_after_insert=*/false);
  run("ATTACH '" + (std::filesystem::path(disk.directory) / "keys.db").string() +
      "' AS native_store");
  run("CREATE TABLE native_store.keys AS SELECT range::BIGINT k FROM range(7)");
  run("CHECKPOINT native_store");
  run("SET gpu_execution=true");
  auto runtime          = sirius::test::get_registered_sirius_context(con);
  auto counters         = runtime->physical_counters();
  counters->track_units = true;
  std::atomic<unsigned> decoded{0}, replays{0};
  counters->parquet_phase_for_testing = [&](std::string const&, bool footer) {
    if (!footer && ++decoded == 3)
      throw sirius::transparent::classified_execution_error(
        sirius::transparent::late_failure_cause::reader_io, "injected DuckLake decode failure");
  };
  runtime->cpu_replay_hook_for_testing = [&] { ++replays; };
  auto clear                           = std::shared_ptr<void>(nullptr, [&](void*) {
    counters->parquet_phase_for_testing  = {};
    runtime->cpu_replay_hook_for_testing = {};
    // The injection happens after worker drains. Release retained test keys before teardown.
    runtime->get_scan_manager().reset_all();
  });
  run("SET enable_duckdb_fallback=true");
  run("SET sirius_test_sync_cpu_replay=true");
  run("SET sirius_test_inject_checkpoint_cleanup_failure=true");
  auto before     = runtime->get_transparent_execution_stats();
  std::string sql = "SELECT q.id,b.k FROM lake.q q JOIN native_store.keys b ON q.value=b.k";
  auto result     = con.Query(
    explicit_entry ? "CALL gpu_execution(" + duckdb::Value(sql).ToSQLString() + ")" : sql);
  REQUIRE(result->HasError());
  INFO(result->GetError());
  CHECK(decoded >= 3);
  CHECK(replays == 0);
  CHECK(runtime->get_transparent_execution_stats().runtime_fallbacks == before.runtime_fallbacks);
  CHECK(runtime->get_runtime_health() == duckdb::SiriusContext::runtime_health::UNAVAILABLE);
  CHECK(runtime->get_scan_manager().checkpoint_key_count() > 0);
}

TEST_CASE_METHOD(
  ducklake_fixture,
  "Production DuckLake SQL does not reuse rows after replacing a file at the same path",
  "[isolated_context][ducklake][ducklake_object_identity]")
{
  if (!sirius::ducklake::bridge_available()) SKIP("DuckLake private binding bridge is disabled");
  attach_lake(con, disk.directory);
  run("SET gpu_execution=false");
  run("CALL lake.set_option('data_inlining_row_limit',0)");
  run("CREATE TABLE lake.q AS SELECT i::BIGINT id,(i%7)::BIGINT value FROM range(2048) r(i)");
  std::string sql = "SELECT value,id FROM lake.q";
  auto original   = ducklake_result_bag(*run(sql));
  run("SET gpu_execution=true");
  auto runtime          = sirius::test::get_registered_sirius_context(con);
  auto counters         = runtime->physical_counters();
  counters->track_units = true;
  std::atomic<unsigned> decoded{0};
  counters->parquet_phase_for_testing = [&](std::string const&, bool footer) {
    if (!footer) ++decoded;
  };
  auto clear =
    std::shared_ptr<void>(nullptr, [&](void*) { counters->parquet_phase_for_testing = {}; });
  run("SET enable_duckdb_fallback=false");
  for (int repeat = 0; repeat < 2; ++repeat)
    CHECK(ducklake_result_bag(*run(sql)) == original);
  REQUIRE(decoded >= 2);
  run("SET gpu_execution=false");
  auto files = run("SELECT data_file FROM ducklake_list_files('lake', 'q')");
  REQUIRE(files->RowCount() == 1);
  auto path        = files->Collection().GetValue(0, 0).ToString();
  auto replacement = (std::filesystem::path(disk.directory) / "replacement.parquet").string();
  run("COPY (SELECT id+10000 AS id,value+100 AS value FROM read_parquet(" +
      duckdb::Value(path).ToSQLString() + ")) TO " + duckdb::Value(replacement).ToSQLString() +
      " (FORMAT PARQUET, FIELD_IDS {id:1,value:2})");
  std::filesystem::rename(replacement, path);
  auto expected = ducklake_result_bag(*run(sql));
  REQUIRE(expected != original);
  run("SET gpu_execution=true");
  auto before = runtime->get_transparent_execution_stats();
  auto result = con.Query(sql);
  if (result->HasError()) {
    // Catalog file evidence may reject external replacement; replay must read the new object.
    run("SET enable_duckdb_fallback=true");
    result = con.Query(sql);
    REQUIRE_FALSE(result->HasError());
    CHECK(runtime->get_transparent_execution_stats().runtime_fallbacks ==
          before.runtime_fallbacks + 1);
  } else {
    CHECK(runtime->get_transparent_execution_stats().executions == before.executions + 1);
    CHECK(runtime->get_transparent_execution_stats().runtime_fallbacks == before.runtime_fallbacks);
  }
  CHECK(ducklake_result_bag(*result) == expected);
}

TEST_CASE_METHOD(ducklake_fixture,
                 "Production DuckLake SQL cancellation waits for its blocked provider query",
                 "[isolated_context][ducklake][ducklake_provider_cancel]")
{
  if (!sirius::ducklake::bridge_available()) SKIP("DuckLake private binding bridge is disabled");
  attach_lake(con, disk.directory);
  run("SET gpu_execution=false");
  run("CALL lake.set_option('data_inlining_row_limit',20000)");
  run("CREATE TABLE lake.q AS SELECT i::BIGINT id,(i%7)::BIGINT value FROM range(4096) r(i)");
  std::string sql = "SELECT value,id FROM lake.q WHERE id%3<>0";
  auto expected   = ducklake_result_bag(*run(sql));
  run("SET gpu_execution=true");
  auto runtime          = sirius::test::get_registered_sirius_context(con);
  auto counters         = runtime->physical_counters();
  counters->track_units = true;
  auto hosts =
    runtime->get_memory_manager().get_memory_spaces_for_tier(cucascade::memory::Tier::HOST);
  REQUIRE(hosts.size() == 1);
  auto reserved = hosts[0]->get_total_reserved_memory();
  run("SET enable_duckdb_fallback=true");
  run("SET sirius_test_sync_cpu_replay=true");
  std::atomic<unsigned> replay_checks{0};
  runtime->cpu_replay_hook_for_testing = [&] { ++replay_checks; };
  query_pause pause;
  std::atomic<bool> first{true};
  std::thread::id execution_owner, provider_owner;
  auto observer     = duckdb::make_shared_ptr<provider_query_observer>();
  observer->observe = [&](duckdb::ClientContext& provider) {
    auto state = duckdb::get_sirius_connection_state(provider);
    // Admission distinguishes reader work from metadata SQL used while binding the user query.
    if (!state || state->provenance() != connection_provenance::provider_internal ||
        hosts[0]->get_total_reserved_memory() <= reserved || !first.exchange(false))
      return;
    provider_owner = std::this_thread::get_id();
    pause.arrive_and_wait(std::chrono::seconds(20));
  };
  duckdb::OptimizerExtension hook;
  hook.optimizer_info        = observer;
  hook.pre_optimize_function = observe_provider_query;
  duckdb::OptimizerExtension::Register(duckdb::DBConfig::GetConfig(*con.context), std::move(hook));
  auto clear   = std::shared_ptr<void>(nullptr, [&](void*) {
    observer->observe                    = {};
    runtime->cpu_replay_hook_for_testing = {};
  });
  auto before  = runtime->get_transparent_execution_stats();
  auto active  = std::async(std::launch::async, [&] {
    execution_owner = std::this_thread::get_id();
    return con.Query(sql);
  });
  bool reached = pause.wait_until_arrived();
  con.Interrupt();
  auto while_blocked    = active.wait_for(std::chrono::milliseconds(50));
  auto held_reservation = hosts[0]->get_total_reserved_memory();
  auto held_scans       = runtime->get_scan_manager().num_active_queries();
  pause.release();
  auto result = active.get();
  REQUIRE(reached);
  CHECK(provider_owner == execution_owner);
  CHECK(while_blocked == std::future_status::timeout);
  CHECK(held_reservation > reserved);
  CHECK(held_scans > 0);
  REQUIRE(result->HasError());
  CHECK(replay_checks == 0);
  CHECK(runtime->get_transparent_execution_stats().runtime_fallbacks == before.runtime_fallbacks);
  result.reset();
  CHECK(hosts[0]->get_total_reserved_memory() == reserved);
  observer->observe = {};
  run("SET enable_duckdb_fallback=false");
  auto next = runtime->get_transparent_execution_stats();
  CHECK(ducklake_result_bag(*run(sql)) == expected);
  CHECK(runtime->get_transparent_execution_stats().executions == next.executions + 1);
}

TEST_CASE("Production DuckLake mixed scans execute and recover across two GPUs",
          "[isolated_context][ducklake_multi_gpu][multi_gpu]")
{
  if (!sirius::ducklake::bridge_available()) SKIP("DuckLake private binding bridge is disabled");
  if (!sirius::test::has_gpus(2)) SKIP("DuckLake device placement requires two GPUs");
  sirius::test::scoped_temp_directory disk;
  auto config_path = write_lake_config(
    disk.directory, [](YAML::Node& config) { config["sirius"]["topology"]["num_gpus"] = 2; });
  sirius::test::shared_test_env env(config_path);
  auto con = env.make_connection();
  if (!load_ducklake(con)) return;
  auto run = [&](std::string const& sql) { return query(con, sql); };
  attach_lake(con, disk.directory);
  run("SET gpu_execution=false");
  run("CALL lake.set_option('data_inlining_row_limit',0)");
  auto use_dv = GENERATE(false, true);
  INFO(use_dv);
  run(std::string("CALL lake.set_option('write_deletion_vectors',") + (use_dv ? "true" : "false") +
      ")");
  run("CREATE TABLE lake.q(id BIGINT,value BIGINT)");
  for (int batch = 0; batch < 16; ++batch) {
    run(
      "INSERT INTO lake.q SELECT i,CASE WHEN i%7=0 THEN NULL ELSE i%17 END "
      "FROM range(" +
      std::to_string(batch * 32768) + "," + std::to_string((batch + 1) * 32768) + ") r(i)");
  }
  run("DELETE FROM lake.q WHERE id%13=0");
  run("ALTER TABLE lake.q RENAME id TO renamed_id");
  run("CALL lake.set_option('data_inlining_row_limit',100)");
  run("INSERT INTO lake.q VALUES (42,2),(42,2),(43,NULL)");
  std::string sql =
    "SELECT value,COUNT(*),SUM(renamed_id) FROM lake.q WHERE renamed_id%3=0 OR renamed_id%7=0 "
    "GROUP BY value";
  auto expected = ducklake_result_bag(*run(sql));
  run("SET gpu_execution=true");
  auto runtime          = sirius::test::get_registered_sirius_context(con);
  auto counters         = runtime->physical_counters();
  counters->track_units = true;
  std::mutex mutex;
  std::set<int> devices;
  bool inject = false, fired = false;
  counters->parquet_phase_for_testing = [&](std::string const&, bool footer) {
    if (footer) return;
    int device = -1;
    if (cudaGetDevice(&device) != cudaSuccess)
      throw std::runtime_error("cannot observe GPU device");
    std::lock_guard lock(mutex);
    devices.insert(device);
    if (inject && devices.size() == 2 && !std::exchange(fired, true))
      throw std::runtime_error("DuckLake test failure after both devices decoded");
  };
  auto clear =
    std::shared_ptr<void>(nullptr, [&](void*) { counters->parquet_phase_for_testing = {}; });
  auto hosts =
    runtime->get_memory_manager().get_memory_spaces_for_tier(cucascade::memory::Tier::HOST);
  REQUIRE(hosts.size() == 1);
  auto reserved = hosts[0]->get_total_reserved_memory();
  for (bool explicit_entry : {false, true}) {
    INFO(explicit_entry);
    for (bool fail : {false, true, false}) {
      INFO(fail);
      devices.clear();
      inject = fail;
      fired  = false;
      run(std::string("SET enable_duckdb_fallback=") + (fail ? "true" : "false"));
      auto before   = runtime->get_transparent_execution_stats();
      auto imported = counters->arrow_rows_imported.load();
      auto result =
        run(explicit_entry ? "CALL gpu_execution(" + duckdb::Value(sql).ToSQLString() + ")" : sql);
      CHECK(ducklake_result_bag(*result) == expected);
      result.reset();
      CHECK(devices.size() == 2);
      CHECK(fired == fail);
      auto after = runtime->get_transparent_execution_stats();
      if (!explicit_entry) {
        CHECK(after.runtime_fallbacks == before.runtime_fallbacks + (fail ? 1 : 0));
        CHECK(after.executions == before.executions + (fail ? 0 : 1));
      }
      if (!fail) CHECK(counters->arrow_rows_imported.load() > imported);
      CHECK(hosts[0]->get_total_reserved_memory() == reserved);
    }
  }
}

TEST_CASE_METHOD(ducklake_fixture,
                 "Production DuckLake inline batch sizing preserves wide nullable rows",
                 "[isolated_context][ducklake][ducklake_inline_batches]")
{
  auto width = GENERATE(2, 5, 32);
  INFO(width);
  if (!sirius::ducklake::bridge_available()) SKIP("DuckLake private binding bridge is disabled");
  attach_lake(con, disk.directory);
  run("SET gpu_execution=false");
  run("CALL lake.set_option('data_inlining_row_limit',20000)");
  std::string fields = "i::BIGINT id";
  for (int col = 1; col < width; ++col) {
    fields += ",CASE WHEN i%13=0 THEN NULL ELSE ";
    fields += col % 3 == 0 ? "i%2=0" : "'value_'||CAST(i%17 AS VARCHAR)";
    fields += " END AS c" + std::to_string(col);
  }
  run("CREATE TABLE lake.q AS SELECT " + fields + " FROM range(4097) r(i)");
  auto expected         = ducklake_result_bag(*run("SELECT * FROM lake.q"));
  auto runtime          = sirius::test::get_registered_sirius_context(con);
  auto counters         = runtime->physical_counters();
  counters->track_units = true;
  run("SET gpu_execution=true");
  run("SET enable_duckdb_fallback=false");
  for (bool explicit_entry : {false, true}) {
    auto batches = counters->arrow_batches_imported.load();
    auto rows    = counters->arrow_rows_imported.load();
    check_gpu_result("SELECT * FROM lake.q", expected, explicit_entry);
    CHECK(counters->arrow_rows_imported.load() - rows == 4097);
    auto imported = counters->arrow_batches_imported.load() - batches;
    CHECK(imported > 0);
    // Provider chunk boundaries affect the exact count.
    // A narrow scan must still need fewer imports than 256-row data batches alone.
    if (width == 2) CHECK(imported < 17);
    if (width == 32) CHECK(imported >= 17);
  }
}

TEST_CASE_METHOD(ducklake_fixture,
                 "DuckLake inline completion does not import empty Arrow batches",
                 "[isolated_context][ducklake][ducklake_inline_completion]")
{
  auto mixed             = GENERATE(false, true);
  auto transaction_local = GENERATE(false, true);
  CAPTURE(mixed, transaction_local);
  if (!sirius::ducklake::bridge_available()) SKIP("DuckLake private binding bridge is disabled");
  query(con, "SET gpu_execution=false");
  attach_lake(con, disk.directory);
  query(con, "CALL lake.set_option('data_inlining_row_limit',100)");
  query(con, "CREATE TABLE lake.q(id BIGINT, s VARCHAR)");
  if (mixed) {
    query(con, "INSERT INTO lake.q VALUES (1,'parquet'),(2,NULL)");
    query(con, "CALL ducklake_flush_inlined_data('lake')");
  }
  if (transaction_local) query(con, "BEGIN");
  query(con, "INSERT INTO lake.q VALUES (3,'inline'),(4,NULL),(NULL,'value')");
  auto expected         = ducklake_result_bag(*query(con, "SELECT * FROM lake.q"));
  auto runtime          = sirius::test::get_registered_sirius_context(con);
  auto counters         = runtime->physical_counters();
  counters->track_units = true;
  query(con, "SET gpu_execution=true");
  query(con, "SET enable_duckdb_fallback=false");
  for (bool explicit_entry : {false, true}) {
    auto prepared = con.Prepare(explicit_entry ? "CALL gpu_execution('SELECT * FROM lake.q')"
                                               : "SELECT * FROM lake.q");
    REQUIRE_FALSE(prepared->HasError());
    for (int repeat = 0; repeat < 3; ++repeat) {
      auto batches = counters->arrow_batches_imported.load();
      auto rows    = counters->arrow_rows_imported.load();
      check_gpu_execution(expected, [&] { return prepared->Execute(); }, explicit_entry);
      CHECK(counters->arrow_rows_imported.load() - rows == 3);
      CHECK(counters->arrow_batches_imported.load() - batches == 1);
    }
  }
  if (transaction_local) query(con, "ROLLBACK");
}

TEST_CASE_METHOD(ducklake_fixture,
                 "DuckLake result fragments execute original bindings without CPU replay",
                 "[isolated_context][ducklake][ducklake_fragment]")
{
  auto shape       = GENERATE(std::string("parquet"),
                        std::string("mixed"),
                        std::string("inline"),
                        std::string("failed_parquet"));
  bool fail_decode = shape == "failed_parquet";
  INFO(shape);
  if (!sirius::ducklake::bridge_available()) SKIP("DuckLake private binding bridge is disabled");
  query(con, "SET gpu_execution=false");
  attach_lake(con, disk.directory);
  query(con, "CALL lake.set_option('data_inlining_row_limit',100)");
  query(con, "CREATE TABLE lake.q(id BIGINT)");
  query(con, "INSERT INTO lake.q VALUES (1),(2),(2)");
  if (shape != "inline") query(con, "CALL ducklake_flush_inlined_data('lake')");
  query(con, "INSERT INTO lake.q VALUES (3),(NULL)");
  if (shape == "parquet" || fail_decode) query(con, "CALL ducklake_flush_inlined_data('lake')");
  auto sql              = std::string("SELECT id FROM lake.q");
  auto expected         = ducklake_result_bag(*query(con, sql));
  auto runtime          = sirius::test::get_registered_sirius_context(con);
  auto counters         = runtime->physical_counters();
  counters->track_units = true;
  con.context->registered_state->Insert(
    sirius::exec::stream_bind_catalog::kStateKey,
    duckdb::make_shared_ptr<sirius::exec::stream_bind_catalog>());
  std::atomic<unsigned> decoded{0};
  counters->parquet_phase_for_testing = [&](std::string const&, bool footer) {
    if (!footer) {
      ++decoded;
      if (fail_decode) throw std::runtime_error("fragment decode failure");
    }
  };
  auto observer          = duckdb::make_shared_ptr<provider_query_observer>();
  unsigned visits        = 0;
  bool inside            = false;
  observer->inspect_plan = [&](duckdb::ClientContext& context, duckdb::LogicalOperator& plan) {
    if (&context != con.context.get() || inside) return;
    inside     = true;
    auto leave = std::shared_ptr<void>(nullptr, [&](void*) { inside = false; });
    ++visits;
    sirius::exec::fragment_spec spec;
    spec.plan_source = [&](duckdb::ClientContext& current) {
      REQUIRE(&current == &context);
      auto prepared = duckdb::make_shared_ptr<duckdb::PreparedStatementData>(
        duckdb::StatementType::SELECT_STATEMENT);
      prepared->types = {duckdb::LogicalType::BIGINT};
      prepared->names = {"id"};
      return sirius::exec::bound_plan{plan.Copy(current), std::move(prepared)};
    };
    sirius::exec::streaming_fragment fragment(context, std::move(spec));
    fragment.build();
    if (fail_decode) {
      REQUIRE_THROWS_WITH(fragment.run(),
                          Catch::Matchers::ContainsSubstring("fragment decode failure"));
      REQUIRE_THROWS(fragment.take_result());
    } else {
      fragment.run();
      CHECK(ducklake_result_bag(*fragment.take_result()) == expected);
    }
  };
  auto clear = std::shared_ptr<void>(nullptr, [&](void*) {
    observer->inspect_plan              = {};
    counters->parquet_phase_for_testing = {};
  });
  duckdb::OptimizerExtension extension;
  extension.optimizer_info    = observer;
  extension.optimize_function = observe_provider_query;
  duckdb::OptimizerExtension::Register(duckdb::DBConfig::GetConfig(*con.context), extension);
  auto imported = counters->arrow_rows_imported.load();
  auto before   = runtime->get_transparent_execution_stats();
  CHECK(ducklake_result_bag(*query(con, sql)) == expected);
  CHECK(visits == 1);
  CHECK((decoded.load() > 0) == (shape != "inline"));
  CHECK((counters->arrow_rows_imported.load() > imported) == (shape != "parquet" && !fail_decode));
  auto after = runtime->get_transparent_execution_stats();
  CHECK(after.fallbacks == before.fallbacks);
  CHECK(after.runtime_fallbacks == before.runtime_fallbacks);
}

TEST_CASE_METHOD(ducklake_fixture,
                 "DuckLake scalar subqueries match CPU results",
                 "[isolated_context][ducklake][ducklake_scalar_subqueries]")
{
  auto parquet = GENERATE(false, true);
  if (!sirius::ducklake::bridge_available()) SKIP("DuckLake private binding bridge is disabled");
  query(con, "SET gpu_execution=false");
  attach_lake(con, disk.directory);
  query(con, "CALL lake.set_option('data_inlining_row_limit',100)");
  query(con, "CREATE TABLE lake.scalar_value AS SELECT 42::BIGINT AS x");
  if (parquet) query(con, "CALL ducklake_flush_inlined_data('lake')");
  for (auto const* sql : {"SELECT (SELECT x FROM lake.scalar_value)",
                          "SELECT (SELECT x FROM lake.scalar_value WHERE x%2=1)"})
    check_gpu_query(sql);
}

TEST_CASE_METHOD(ducklake_fixture,
                 "DuckLake preserves discovery errors when GPU fallback is refused",
                 "[isolated_context][ducklake][ducklake_discovery_error]")
{
  query(con, "SET gpu_execution=false");
  auto attach = "ATTACH 'ducklake:" + disk.directory + "/meta.db' AS lake (DATA_PATH '" +
                disk.directory + "/data', METADATA_CATALOG 'meta', DATA_INLINING_ROW_LIMIT 0)";
  query(con, attach);
  query(con, "CREATE TABLE lake.t(id INTEGER)");
  query(con, "INSERT INTO lake.t VALUES (1), (2)");
  query(con, "DELETE FROM lake.t WHERE id=2");
  std::vector<std::string> statements{
    "SELECT * FROM lake.t",
    "SELECT * FROM ducklake_table_insertions('lake','main','t',0,2)",
    "SELECT * FROM ducklake_table_deletions('lake','main','t',2,3)"};
  for (auto const& sql : statements) {
    query(con, "SET gpu_execution=false");
    auto expected = ducklake_result_bag(*query(con, sql));
    query(con, "SET gpu_execution=true");
    query(con, "SET enable_duckdb_fallback=true");
    CHECK(ducklake_result_bag(*query(con, sql)) == expected);
  }
  query(con, "SET gpu_execution=false");
  query(con, "UPDATE meta.ducklake_data_file SET file_format='foo'");
  query(con, "DETACH lake");
  query(con, attach);
  for (auto const& sql : statements) {
    query(con, "SET gpu_execution=false");
    auto cpu = con.Query(sql);
    REQUIRE(cpu->HasError());
    REQUIRE(cpu->GetError().find("Unsupported data file format: foo") != std::string::npos);
    query(con, "SET gpu_execution=true");
    for (bool fallback : {true, false}) {
      query(con, std::string("SET enable_duckdb_fallback=") + (fallback ? "true" : "false"));
      for (auto const& statement :
           {sql, "CALL gpu_execution(" + duckdb::Value(sql).ToSQLString() + ")"}) {
        auto result = con.Query(statement);
        REQUIRE(result->HasError());
        INFO(result->GetError());
        if (fallback) {
          CHECK(result->GetError().find("Unsupported data file format: foo") != std::string::npos);
          CHECK(result->GetError().find("source discovery incomplete") == std::string::npos);
        } else {
          CHECK((result->GetError().find("GPU") != std::string::npos ||
                 result->GetError().find("SiriusExecuteQuery error:") != std::string::npos));
        }
      }
    }
  }
}

TEST_CASE_METHOD(ducklake_fixture,
                 "DuckLake reuses delete preparation capacity across files and scans",
                 "[isolated_context][ducklake][ducklake_delete_budget]")
{
  auto files = GENERATE(1, 2, 8);
  INFO(files);
  if (!sirius::ducklake::bridge_available()) SKIP("DuckLake private binding bridge is disabled");
  run("SET gpu_execution=false");
  attach_lake(con, disk.directory);
  run("CALL lake.set_option('data_inlining_row_limit',10)");
  run("CREATE TABLE lake.q(i BIGINT)");
  for (int file = 0; file < files; ++file) {
    run("INSERT INTO lake.q SELECT i FROM range(" + std::to_string(file * 100) + "," +
        std::to_string((file + 1) * 100) + ") t(i)");
  }
  run("DELETE FROM lake.q WHERE i%100=0");
  run("INSERT INTO lake.q VALUES (-1),(-2)");
  std::vector<std::string> sql{"SELECT count(*),sum(i) FROM lake.q",
                               "SELECT count(*) FROM lake.q",
                               "SELECT i FROM lake.q WHERE i=50 OR i%10=9",
                               "SELECT count(*),sum(a.i) FROM lake.q a JOIN lake.q b ON a.i=b.i"};
  std::vector<std::vector<std::string>> expected;
  for (auto const& text : sql)
    expected.push_back(ducklake_result_bag(*run(text)));
  auto runtime  = sirius::test::get_registered_sirius_context(con);
  auto observed = std::make_shared<observed_host_reservations>(runtime->get_memory_manager());
  runtime->physical_counters()->track_units                      = true;
  runtime->physical_counters()->preparation_provider_for_testing = observed;
  runtime->physical_counters()->iceberg_statement_route_for_testing =
    [&](std::string_view, sirius::scan_manager::admission_decision const& decision) {
      if (!decision.deferred) return;
      std::lock_guard lock(observed->seen->mutex);
      observed->seen->exceeded |=
        decision.sigma_retained + decision.n_permits * decision.w > decision.c_obtained ||
        decision.n_permits < 1;
    };
  auto clear_provider = std::shared_ptr<void>(nullptr, [&](void*) {
    runtime->physical_counters()->preparation_provider_for_testing.reset();
    runtime->physical_counters()->iceberg_statement_route_for_testing = {};
  });
  auto hosts =
    runtime->get_memory_manager().get_memory_spaces_for_tier(cucascade::memory::Tier::HOST);
  REQUIRE(hosts.size() == 1);
  auto const reserved = hosts[0]->get_total_reserved_memory();
  run("SET gpu_execution=true");
  run("SET enable_duckdb_fallback=false");
  for (int repeat = 0; repeat < 3; ++repeat) {
    for (size_t i = 0; i < sql.size(); ++i) {
      INFO(sql[i]);
      check_gpu_result(sql[i], expected[i]);
      CHECK(hosts[0]->get_total_reserved_memory() == reserved);
    }
  }
  if (files > 1) {
    auto counters = runtime->physical_counters();
    std::atomic<unsigned> decoded{0};
    counters->parquet_phase_for_testing = [&](std::string const&, bool footer) {
      if (!footer && ++decoded == 2) throw std::runtime_error("delete visibility decode failure");
    };
    auto clear =
      std::shared_ptr<void>(nullptr, [&](void*) { counters->parquet_phase_for_testing = {}; });
    run("SET enable_duckdb_fallback=true");
    auto before = get_transparent_execution_stats(con);
    CHECK(ducklake_result_bag(*run(sql[0])) == expected[0]);
    CHECK(decoded.load() >= 2);
    CHECK(get_transparent_execution_stats(con).runtime_fallbacks == before.runtime_fallbacks + 1);
    CHECK(hosts[0]->get_total_reserved_memory() == reserved);
    clear.reset();
    run("SET enable_duckdb_fallback=false");
    CHECK(ducklake_result_bag(*run(sql[0])) == expected[0]);
    CHECK(hosts[0]->get_total_reserved_memory() == reserved);
  }
  CHECK_FALSE(observed->seen->exceeded);
  CHECK(observed->seen->peak_live > 0);
  CHECK(observed->seen->live == 0);
  CHECK(observed->seen->grants == 0);
}

TEST_CASE_METHOD(ducklake_fixture,
                 "DuckLake empty scans preserve scalar aggregate results",
                 "[isolated_context][ducklake][ducklake_empty_aggregate]")
{
  if (!sirius::ducklake::bridge_available()) SKIP("DuckLake private binding bridge is disabled");
  auto shape = GENERATE(
    std::string("empty"), std::string("inline"), std::string("parquet"), std::string("mixed"));
  INFO(shape);
  query(con, "SET gpu_execution=false");
  attach_lake(con, disk.directory);
  query(con, "CALL lake.set_option('data_inlining_row_limit',100)");
  query(con, "CREATE TABLE lake.q(x BIGINT, s VARCHAR)");
  if (shape != "empty") query(con, "INSERT INTO lake.q VALUES (NULL,NULL),(42,'value')");
  if (shape == "parquet" || shape == "mixed")
    query(con, "CALL ducklake_flush_inlined_data('lake')");
  if (shape == "mixed") query(con, "INSERT INTO lake.q VALUES (NULL,NULL)");
  for (auto suffix : {"", " WHERE x>1000", " WHERE x%3=1"}) {
    if (shape == "empty" && *suffix) continue;
    auto sql =
      std::string(
        "SELECT COUNT(*),COUNT(x),SUM(x),FIRST(x),FIRST(s),AVG(x),MIN(x),MAX(x) FROM lake.q") +
      suffix;
    CAPTURE(sql);
    auto expected = cpu_result(sql);
    auto prepared = con.Prepare(sql);
    REQUIRE_FALSE(prepared->HasError());
    for (int repeat = 0; repeat < 2; ++repeat)
      check_gpu_execution(expected, [&] { return prepared->Execute(); });
  }
}

TEST_CASE_METHOD(ducklake_fixture,
                 "DuckLake inline projection reads only required columns",
                 "[isolated_context][ducklake][ducklake_projection]")
{
  if (!sirius::ducklake::bridge_available()) SKIP("DuckLake private binding bridge is disabled");
  auto local = GENERATE(false, true);
  CAPTURE(local);
  query(con, "SET gpu_execution=false");
  attach_lake(con, disk.directory);
  query(con, "CALL lake.set_option('data_inlining_row_limit', 20000)");
  query(con, "CREATE TABLE lake.q(id BIGINT, unused VARCHAR, label VARCHAR, flag BOOLEAN)");
  if (local) query(con, "BEGIN");
  query(con,
        "INSERT INTO lake.q SELECT i, repeat('unused', 16), "
        "CASE WHEN i%7=0 THEN NULL ELSE 'value_'||CAST(i%13 AS VARCHAR) END, i%2=0 "
        "FROM range(4097) r(i)");
  auto observer = duckdb::make_shared_ptr<provider_query_observer>();
  // Keep COUNT on the scan path instead of replacing it with catalog statistics.
  sirius::test::disabled_optimizers_guard const scan_count(con, "statistics_propagation");
  std::vector<size_t> read_widths;

  observer->inspect_plan = [&](duckdb::ClientContext& context, duckdb::LogicalOperator& plan) {
    if (&context != con.context.get() &&
        context.GetCurrentQuery().find(" inlined_data") != std::string::npos)
      read_widths.push_back(plan.GetColumnBindings().size());
  };
  duckdb::OptimizerExtension hook;
  hook.optimizer_info        = observer;
  hook.pre_optimize_function = observe_provider_query;
  duckdb::OptimizerExtension::Register(duckdb::DBConfig::GetConfig(*con.context), std::move(hook));
  auto clear = std::shared_ptr<void>(nullptr, [&](void*) { observer->inspect_plan = {}; });
  for (auto const& [sql, width] : std::vector<std::pair<std::string, size_t>>{
         {"SELECT id FROM lake.q", 1},
         {"SELECT COUNT(*) FROM lake.q", 1},
         {"SELECT label,id,label FROM lake.q WHERE flag AND id>10", 3},
         {"SELECT id FROM lake.q WHERE label='value_2'", 2}}) {
    read_widths.clear();
    check_gpu_query(sql);
    if (!local) REQUIRE_FALSE(read_widths.empty());
    for (auto read_width : read_widths)
      CHECK(read_width <= width);
  }
  if (local) query(con, "ROLLBACK");
}

TEST_CASE_METHOD(ducklake_fixture,
                 "DuckLake validates local Parquet deletes across parallel row group boundaries",
                 "[ducklake][ducklake_parallel_deletes][isolated_context]")
{
  if (!sirius::ducklake::bridge_available()) SKIP("DuckLake private binding bridge is disabled");
  run("SET gpu_execution=false");
  run("SET threads=1");
  attach_lake(con, disk.directory);
  run("CALL lake.set_option('data_inlining_row_limit',0)");
  run("CALL lake.set_option('write_deletion_vectors',false)");
  run("CREATE TABLE lake.q AS SELECT i::BIGINT id FROM range(16384) t(i)");
  run("DELETE FROM lake.q WHERE id % 2 = 0");
  auto files = run("SELECT delete_file FROM ducklake_list_files('lake','q')");
  REQUIRE(files->RowCount() == 1);
  auto source           = files->Collection().GetValue(0, 0).ToSQLString();
  const std::string sql = "SELECT sum(id) FROM lake.q";
  auto expected         = ducklake_result_bag(*run(sql));
  for (int delta : {0, 2, 4}) {
    CAPTURE(delta);
    run("SET gpu_execution=false");
    run("SET threads=1");
    auto path =
      std::filesystem::path(disk.directory) / ("deletes-" + std::to_string(delta) + ".parquet");
    auto dest       = duckdb::Value(path.string()).ToSQLString();
    std::string pos = "CASE WHEN pos=4096 THEN pos-" + std::to_string(delta) + " ELSE pos END";
    run("COPY (SELECT file_path," + pos + " AS pos FROM read_parquet(" + source +
        ") t ORDER BY t.pos) TO " + dest + " (FORMAT PARQUET, ROW_GROUP_SIZE 2048)");
    REQUIRE(run("SELECT count(DISTINCT row_group_id) FROM parquet_metadata(" + dest + ")")
              ->Collection()
              .GetValue(0, 0)
              .GetValue<int64_t>() == 4);
    run("UPDATE __ducklake_metadata_lake.ducklake_delete_file SET path=" + dest +
        ",path_is_relative=false,footer_size=NULL,file_size_bytes=" +
        std::to_string(std::filesystem::file_size(path)));
    run("SET threads=4");
    run("SET enable_duckdb_fallback=false");
    run("SET gpu_execution=true");
    if (delta) {
      auto result = con.Query(sql);
      REQUIRE(result);
      REQUIRE(result->HasError());
      CHECK(result->GetError().find("strictly increasing") != std::string::npos);
    } else {
      check_gpu_result(sql, expected);
    }
  }
}

TEST_CASE_METHOD(ducklake_fixture,
                 "Interrupting a DuckLake Parquet delete read preserves cancellation",
                 "[isolated_context][ducklake][ducklake_delete_interrupt]")
{
  if (!sirius::ducklake::bridge_available()) SKIP("DuckLake private binding bridge is disabled");
  run("SET gpu_execution=false");
  attach_lake(con, disk.directory);
  run("CALL lake.set_option('data_inlining_row_limit',0)");
  run("CALL lake.set_option('write_deletion_vectors',false)");
  run("CREATE TABLE lake.q AS SELECT range id FROM range(4096)");
  run("DELETE FROM lake.q WHERE id < 1234");
  run("SET gpu_execution=true");
  run("SET enable_duckdb_fallback=true");
  run("SET sirius_test_sync_cpu_replay=true");
  auto runtime  = sirius::test::get_registered_sirius_context(con);
  auto counters = runtime->physical_counters();
  auto observed = std::make_shared<observed_host_reservations>(runtime->get_memory_manager());
  std::atomic<unsigned> interrupts{0}, replays{0};
  // The delete reader has bound the footer and is reserving its position vector.
  observed->seen->on_allocate = [&](uint64_t bytes) {
    if (bytes == 1234 * sizeof(int64_t)) {
      ++interrupts;
      con.Interrupt();
    }
  };
  counters->track_units                      = true;
  counters->preparation_provider_for_testing = observed;
  runtime->cpu_replay_hook_for_testing       = [&] { ++replays; };
  auto cleanup                               = std::shared_ptr<void>(nullptr, [&](void*) {
    counters->preparation_provider_for_testing.reset();
    runtime->cpu_replay_hook_for_testing = {};
  });
  SECTION("the query returns cancellation without replay")
  {
    auto result = con.Query("SELECT sum(id) FROM lake.q");
    REQUIRE(interrupts == 1);
    REQUIRE(result->HasError());
    INFO(result->GetError());
    CHECK(result->GetErrorObject().Type() == duckdb::ExceptionType::INTERRUPT);
    CHECK(replays == 0);
  }
  SECTION("the delete preparation job preserves the original exception")
  {
    namespace dl = sirius::ducklake;
    namespace sm = sirius::scan_manager;
    run("SET gpu_execution=false");
    auto observer          = duckdb::make_shared_ptr<provider_query_observer>();
    bool interrupted_job   = false;
    observer->inspect_plan = [&](duckdb::ClientContext& ctx, duckdb::LogicalOperator& plan) {
      if (&ctx != con.context.get()) return;
      auto visit = [&](auto&& self, duckdb::LogicalOperator& node) -> void {
        if (node.type == duckdb::LogicalOperatorType::LOGICAL_GET) {
          auto& get = node.Cast<duckdb::LogicalGet>();
          if (get.function.GetName() == "ducklake_scan") {
            auto binding = dl::binding_handle::capture(get.function, get.bind_data.get(), ctx);
            binding->bind_execution_owner();
            binding->configure_scan({0}, nullptr);
            auto budget = std::make_unique<sm::preparation_ledger>(*observed);
            std::vector<sm::scan_envelope> envelopes{
              {{dl::scan_ingestible::envelope(1, 0)}, {}, true, 1}};
            auto decision = budget->admit(envelopes, observed->space);
            auto admission =
              std::make_shared<sm::preparation_admission>(std::move(budget), decision);
            auto source = binding->make_parquet_unit(0, 1, 1, admission, {0});
            source->attach_preparation_ledger(
              std::shared_ptr<sm::preparation_ledger>(admission->consume()));
            auto lanes = source->make_preparation_sources({});
            auto claim = lanes.front().try_claim();
            auto job   = std::get<sm::preparation_job>(std::move(claim));
            try {
              job.run({});
            } catch (duckdb::InterruptException const&) {
              interrupted_job = true;
              throw;
            }
          }
        }
        for (auto& child : node.children)
          self(self, *child);
      };
      visit(visit, plan);
    };
    duckdb::OptimizerExtension hook;
    hook.optimizer_info        = observer;
    hook.pre_optimize_function = observe_provider_query;
    duckdb::OptimizerExtension::Register(duckdb::DBConfig::GetConfig(*con.context),
                                         std::move(hook));
    auto detach = std::shared_ptr<void>(nullptr, [&](void*) { observer->inspect_plan = {}; });
    auto result = con.Query("SELECT sum(id) FROM lake.q");
    REQUIRE(result->HasError());
    REQUIRE(interrupts == 1);
    CHECK(interrupted_job);
  }
  CHECK(observed->seen->live == 0);
  CHECK(observed->seen->grants == 0);
}

TEST_CASE_METHOD(ducklake_fixture,
                 "DuckLake lookup rejects foreign transaction and reader bindings",
                 "[ducklake][ducklake_binding_lookup][isolated_context]")
{
  if (!sirius::ducklake::bridge_available()) SKIP("DuckLake private binding bridge is disabled");
  run("SET gpu_execution=false");
  attach_lake(con, disk.directory);
  run("CREATE TABLE lake.q(i INTEGER)");
  auto sibling = env.make_connection();
  query(sibling, "SET gpu_execution=false");
  run("BEGIN");
  query(sibling, "BEGIN");
  auto first = con.ExtractPlan("SELECT i FROM lake.q");
  auto same  = con.ExtractPlan("SELECT i FROM lake.q");
  auto other = sibling.ExtractPlan("SELECT i FROM lake.q");
  auto* a    = ducklake_get(*first);
  auto* b    = ducklake_get(*same);
  auto* c    = ducklake_get(*other);
  REQUIRE(a);
  REQUIRE(b);
  REQUIRE(c);
  using sirius::planner::lookup_connector;
  REQUIRE(lookup_connector(*a, *con.context));
  REQUIRE(lookup_connector(*b, *con.context));
  REQUIRE(lookup_connector(*c, *sibling.context));
  using sirius::planner::lookup_connector_classified;
  using sirius::planner::lookup_decline;
  CHECK(lookup_connector_classified(*a, *sibling.context).decline ==
        lookup_decline::bind_data_mismatch);

  REQUIRE(a->function.function_info.get() != b->function.function_info.get());
  // Keep the reader and its original transaction; change only the function-info owner.
  std::swap(a->function.function_info, b->function.function_info);
  auto mismatched = lookup_connector_classified(*a, *con.context);
  std::swap(a->function.function_info, b->function.function_info);
  CHECK(mismatched.decline == lookup_decline::bind_data_mismatch);
}

TEST_CASE_METHOD(ducklake_fixture,
                 "Normal EOS retires an in-flight DuckLake Parquet metadata job",
                 "[ducklake][ducklake_parquet_eos][isolated_context]")
{
  namespace dl   = sirius::ducklake;
  namespace sm   = sirius::scan_manager;
  namespace prep = sirius::test::preparation;
  if (!dl::bridge_available()) SKIP("DuckLake private binding bridge is disabled");
  run("SET gpu_execution=false");
  attach_lake(con, disk.directory);
  run("CALL lake.set_option('data_inlining_row_limit',0)");
  run("CREATE TABLE lake.q AS SELECT 42 AS i");
  auto observer          = duckdb::make_shared_ptr<provider_query_observer>();
  bool exercised         = false;
  observer->inspect_plan = [&](duckdb::ClientContext& ctx, duckdb::LogicalOperator& plan) {
    if (&ctx != con.context.get()) return;
    auto* get = ducklake_get(plan);
    if (!get) return;
    auto binding = dl::binding_handle::capture(get->function, get->bind_data.get(), ctx);
    binding->bind_execution_owner();
    binding->configure_scan({0}, nullptr);
    sm::test::test_reservation_provider backing;
    auto budget = std::make_unique<sm::preparation_ledger>(backing);
    auto env    = dl::parquet_unit_ingestible::envelope(binding->unit_at(0).value());
    std::vector<sm::scan_envelope> scans{{{env}, {}, true, 1}};
    backing.grant_bytes = std::max<uint64_t>(1, env.retained + env.temporary_charge(1));
    auto decision  = budget->admit(scans, sm::memory_space_id{cucascade::memory::Tier::HOST, 0});
    auto admission = std::make_shared<sm::preparation_admission>(std::move(budget), decision);
    auto ledger    = std::shared_ptr<sm::preparation_ledger>(admission->consume());
    auto parquet   = binding->make_parquet_unit(0, 1, 1, admission, {0});
    parquet->attach_preparation_ledger(ledger);
    auto claim =
      parquet->try_claim_split([](std::string_view) -> std::shared_ptr<cucascade::io::ioctx> {
        throw std::logic_error("stopped job reached Parquet I/O");
      });
    auto job = std::get<sm::preparation_job>(std::move(claim));
    prep::fixture f;
    auto hold  = f.make_hold();
    auto out   = std::make_shared<prep::sink>();
    auto guard = f.shutdown_guard();
    f.source(std::make_shared<prep::accumulator>(),
             out,
             [claimed = false,
              work    = std::move(job),
              hold]() mutable -> sm::preparation_coordinator::claim_result {
               if (std::exchange(claimed, true)) return {};
               return sm::preparation_job{
                 [job = std::move(work), hold](sm::preparation_stop_token stop) mutable {
                   hold->wait();
                   return job.run(stop);
                 }};
             });
    f.start();
    REQUIRE(hold->await());
    f.coordinator.request_stop(sm::stop_reason::normal_eos);
    hold->release();
    f.finish();
    CHECK_FALSE(f.completion.has_error());
    CHECK_FALSE(f.owner_error);
    CHECK(ledger->permits_in_flight() == 0);
    exercised = true;
  };
  duckdb::OptimizerExtension hook;
  hook.optimizer_info        = observer;
  hook.pre_optimize_function = observe_provider_query;
  duckdb::OptimizerExtension::Register(duckdb::DBConfig::GetConfig(*con.context), std::move(hook));
  auto detach = std::shared_ptr<void>(nullptr, [&](void*) { observer->inspect_plan = {}; });
  run("SELECT i FROM lake.q");
  CHECK(exercised);
}

TEST_CASE_METHOD(ducklake_fixture,
                 "DuckLake decimals preserve precision across inline Parquet and mixed scans",
                 "[ducklake][ducklake_decimal][isolated_context]")
{
  if (!sirius::ducklake::bridge_available()) SKIP("DuckLake bridge is disabled");
  auto mode       = GENERATE(std::string("inline"), std::string("parquet"), std::string("mixed"));
  auto precision  = GENERATE(5, 9, 10, 18, 19, 38);
  auto scale_mode = GENERATE(0, 1, 2);
  auto scale      = scale_mode == 0 ? 0 : scale_mode == 1 ? 2 : precision;
  CAPTURE(mode, precision, scale);
  attach_lake(con, disk.directory);
  run("SET gpu_execution=false");
  auto limit = mode == "inline" ? 10000 : mode == "parquet" ? 0 : 10;
  run("CALL lake.set_option('data_inlining_row_limit'," + std::to_string(limit) + ")");
  auto type    = "DECIMAL(" + std::to_string(precision) + "," + std::to_string(scale) + ")";
  auto maximum = (precision == scale ? std::string("0") : std::string(precision - scale, '9')) +
                 (scale ? "." + std::string(scale, '9') : "");
  std::string small = precision == scale ? "0.23" : "1.23";
  run("CREATE TABLE lake.decimals(id BIGINT, value " + type + ")");
  run("INSERT INTO lake.decimals SELECT i, CASE i%4 WHEN 0 THEN '" + maximum + "' WHEN 1 THEN '-" +
      maximum + "' WHEN 2 THEN '0' ELSE NULL END::" + type + " FROM range(4097) t(i)");
  run("ALTER TABLE lake.decimals ADD COLUMN initial " + type + " DEFAULT -" + small);
  run("ALTER TABLE lake.decimals ADD COLUMN missing " + type);
  run("INSERT INTO lake.decimals VALUES (5000, " + small + ", -" + small + ", NULL), (5001, -" +
      small + ", -" + small + ", NULL)");
  auto counters         = sirius::test::get_registered_sirius_context(con)->physical_counters();
  counters->track_units = true;
  auto inline_rows      = counters->arrow_rows_imported.load();
  check_gpu_query("SELECT id, value, initial, missing FROM lake.decimals");
  CHECK(counters->arrow_rows_imported.load() - inline_rows == (mode == "inline"  ? 4099
                                                               : mode == "mixed" ? 2
                                                                                 : 0));
  check_gpu_query("SELECT id, value FROM lake.decimals WHERE value > 0::" + type +
                  " OR value IS NULL");
  check_gpu_query("SELECT count(*), min(value), max(value) FROM lake.decimals");
  if (scale != precision) check_gpu_query("SELECT sum(initial) FROM lake.decimals");
  check_gpu_query("SELECT initial, missing FROM lake.decimals WHERE id < 4097");
  // Exercise lifecycle behavior once per physical width and storage mode.
  if (scale == 2 && (precision == 9 || precision == 18 || precision == 38)) {
    check_gpu_query("SELECT value, initial FROM lake.decimals WHERE id < 0");
    auto expected = cpu_result("SELECT value, initial FROM lake.decimals WHERE id >= 5000");
    run("PREPARE decimal_query AS SELECT value, initial FROM lake.decimals WHERE id >= 5000");
    check_gpu_result("EXECUTE decimal_query", expected);
    check_gpu_result("EXECUTE decimal_query", expected);
  }
}

TEST_CASE_METHOD(ducklake_fixture,
                 "DuckLake wide decimal inline batches fit their preparation envelope",
                 "[ducklake][ducklake_decimal][isolated_context]")
{
  if (!sirius::ducklake::bridge_available()) SKIP("DuckLake bridge is disabled");
  attach_lake(con, disk.directory);
  run("SET gpu_execution=false");
  run("CALL lake.set_option('data_inlining_row_limit',10000)");
  std::string columns;
  for (int i = 0; i < 31; ++i) {
    if (i) columns += ",";
    columns +=
      "CASE WHEN i%3=0 THEN NULL ELSE '-12345678901234567890.1234567890'::DECIMAL(38,10) END AS d" +
      std::to_string(i);
  }
  columns += ",CASE WHEN i%2=0 THEN 'decimal payload' ELSE NULL END AS label";
  run("CREATE TABLE lake.wide AS SELECT " + columns + " FROM range(601) t(i)");
  auto counters         = sirius::test::get_registered_sirius_context(con)->physical_counters();
  counters->track_units = true;
  auto inline_rows      = counters->arrow_rows_imported.load();
  check_gpu_query("SELECT * FROM lake.wide");
  CHECK(counters->arrow_rows_imported.load() - inline_rows == 601);
  check_gpu_query("SELECT d0, d30, label FROM lake.wide WHERE d15 IS NOT NULL");
}

namespace {
auto snapshot_config()
{
  if (!sirius::ducklake::bridge_available()) SKIP("DuckLake private binding bridge is disabled");
  return lake_config();
}
struct snapshot_fixture : ducklake_fixture {
  snapshot_fixture() : ducklake_fixture(snapshot_config()) {}
  void check_pruned_delete_positions(bool remote);
};
}  // namespace

TEST_CASE_METHOD(
  snapshot_fixture,
  "DuckLake snapshot visibility native compaction and snapshot deletes match CPU on GPU",
  "[ducklake_snapshot_visibility_gpu][isolated_context]")
{
  auto vectors     = GENERATE(false, true);
  auto interleaved = GENERATE(false, true);
  INFO("deletion vectors: " << vectors << ", interleaved: " << interleaved);
  query(con, "SET gpu_execution=false");
  attach_lake(con, disk.directory);
  query(con, "CALL lake.set_option('data_inlining_row_limit', 0)");
  query(con, "CALL lake.set_option('parquet_row_group_size', 2048)");
  query(con,
        std::string("CALL lake.set_option('write_deletion_vectors', ") +
          (vectors ? "true)" : "false)"));
  query(con, "CREATE TABLE lake.q(id BIGINT, value BIGINT)");
  std::vector<std::string> versions;
  auto remember = [&] {
    versions.push_back(query(con, "SELECT max(snapshot_id) FROM ducklake_snapshots('lake')")
                         ->Collection()
                         .GetValue(0, 0)
                         .ToString());
  };
  query(con, "INSERT INTO lake.q SELECT i, i%7 FROM range(4096) r(i)");
  remember();
  if (interleaved) {
    query(con, "DELETE FROM lake.q WHERE id=1");
    remember();
  }
  query(con, "INSERT INTO lake.q SELECT i, i%7 FROM range(4096,8192) r(i)");
  remember();
  if (interleaved) {
    // The provider excludes files with deletes from compaction. Keep deleted A
    // and merge clean B/C: historical reads combine A's snapshot deletes with
    // B/C's per-row insertion visibility in the same bound table.
    query(con, "INSERT INTO lake.q SELECT i, i%7 FROM range(8192,12288) r(i)");
    remember();
  }
  auto merged = query(con, "CALL ducklake_merge_adjacent_files('lake', 'q')");
  REQUIRE(merged->RowCount() == 1);
  REQUIRE(query(con,
                "SELECT count(*) FROM __ducklake_metadata_lake.ducklake_data_file "
                "WHERE end_snapshot IS NULL AND partial_max IS NOT NULL")
            ->Collection()
            .GetValue(0, 0)
            .GetValue<int64_t>() == 1);
  for (int id : {1, 2051, 6145}) {
    if (interleaved && id == 1) continue;
    query(con, "DELETE FROM lake.q WHERE id=" + std::to_string(id));
    remember();
  }
  auto deletes = query(con,
                       "SELECT format, partial_max FROM "
                       "__ducklake_metadata_lake.ducklake_delete_file WHERE end_snapshot IS NULL");
  REQUIRE(deletes->RowCount() == (interleaved ? 2 : 1));
  bool snapshot_delete = false;
  for (duckdb::idx_t row = 0; row < deletes->RowCount(); ++row) {
    REQUIRE(deletes->Collection().GetValue(0, row).ToString() == (vectors ? "puffin" : "parquet"));
    snapshot_delete |= !deletes->Collection().GetValue(1, row).IsNull();
  }
  REQUIRE(snapshot_delete);
  auto runtime = con.context->registered_state->Get<duckdb::SiriusContext>("sirius_state");
  REQUIRE(runtime);
  auto counters         = runtime->physical_counters();
  counters->track_units = true;
  std::atomic<unsigned> decoded{0};
  counters->parquet_phase_for_testing = [&](std::string const&, bool footer) {
    if (!footer) ++decoded;
  };
  auto clear = std::shared_ptr<void>(nullptr, [&](void*) {
    counters->parquet_phase_for_testing  = {};
    counters->parquet_filter_for_testing = {};
  });
  query(con, "SET enable_duckdb_fallback=false");
  query(con, "SET disabled_optimizers='statistics_propagation'");
  for (auto const& version : versions) {
    auto table      = "lake.q AT (VERSION => " + version + ")";
    auto pruned_sql = "SELECT id FROM " + table + " WHERE id>=6144 AND id<8192";
    for (auto const& sql :
         {"SELECT id FROM " + table,
          "SELECT 1 FROM " + table,
          "SELECT a.id,b.value FROM (SELECT * FROM " + table + ") a JOIN (SELECT * FROM " + table +
            ") b ON a.id=b.id WHERE a.id%17=0",
          "SELECT value,id FROM " + table + " WHERE id%3=0",
          "SELECT 1 FROM " + table + " WHERE id%5=0",
          "SELECT id FROM " + table + " WHERE id=99999",
          pruned_sql}) {
      INFO(sql);
      query(con, "SET gpu_execution=false");
      auto expected = ducklake_result_bag(*query(con, sql));
      query(con, "SET gpu_execution=true");
      for (auto const& statement :
           {sql, "CALL gpu_execution(" + duckdb::Value(sql).ToSQLString() + ")"}) {
        auto before = runtime->get_transparent_execution_stats();
        auto reads  = decoded.load();
        CHECK(ducklake_result_bag(*query(con, statement)) == expected);
        auto after = runtime->get_transparent_execution_stats();
        CHECK(after.fallbacks == before.fallbacks);
        CHECK(after.runtime_fallbacks == before.runtime_fallbacks);
        // Empty/pruned queries can legitimately skip all data pages.
        if (!expected.empty()) { CHECK(decoded.load() > reads); }
      }
    }
  }
}

void snapshot_fixture::check_pruned_delete_positions(bool remote)
{
  auto vectors = GENERATE(false, true);
  INFO("deletion vectors: " << vectors);
  query(con, "SET gpu_execution=false");
  attach_lake(con, disk.directory);
  query(con, "CALL lake.set_option('data_inlining_row_limit', 0)");
  query(con,
        std::string("CALL lake.set_option('write_deletion_vectors', ") +
          (vectors ? "true)" : "false)"));
  query(con, "CREATE TABLE lake.q(id BIGINT)");
  std::vector<std::string> versions;
  auto remember = [&] {
    versions.push_back(query(con, "SELECT max(snapshot_id) FROM ducklake_snapshots('lake')")
                         ->Collection()
                         .GetValue(0, 0)
                         .ToString());
  };
  // Ordered compaction forces the provider's default 122880-row groups.
  // A spans two groups; B adds a third with a later insertion snapshot.
  query(con, "INSERT INTO lake.q SELECT i FROM range(245760) r(i)");
  remember();
  query(con, "INSERT INTO lake.q SELECT i FROM range(245760,368640) r(i)");
  remember();
  REQUIRE(query(con, "CALL ducklake_merge_adjacent_files('lake', 'q')")->RowCount() == 1);
  query(con, "DELETE FROM lake.q WHERE id=123001");
  remember();
  query(con, "DELETE FROM lake.q WHERE id=123002");
  remember();
  REQUIRE(query(con,
                "SELECT count(*) FROM __ducklake_metadata_lake.ducklake_delete_file "
                "WHERE end_snapshot IS NULL AND partial_max IS NOT NULL")
            ->Collection()
            .GetValue(0, 0)
            .GetValue<int64_t>() == 1);
  std::vector<std::string> statements;
  std::vector<std::vector<std::string>> expected;
  std::array<size_t, 4> expected_sizes{11, 11, 10, 9};
  for (size_t i = 0; i < versions.size(); ++i) {
    statements.push_back("SELECT id FROM lake.q AT (VERSION => " + versions[i] +
                         ") WHERE id>=123000 AND id<123011");
    expected.push_back(ducklake_result_bag(*query(con, statements.back())));
    REQUIRE(expected.back().size() == expected_sizes[i]);
  }
  // Preserve the native local oracle before changing only the delete locator.
  std::unique_ptr<cucascade::test::loopback_range_server> server;
  if (remote) {
    auto files = query(con, "SELECT delete_file FROM ducklake_list_files('lake','q')");
    REQUIRE(files->RowCount() == 1);
    std::ifstream input(files->Collection().GetValue(0, 0).ToString(), std::ios::binary);
    REQUIRE(input.good());
    std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(input)), {});
    cucascade::test::range_fault_policy policy;
    policy.successful_head_etag = policy.successful_get_etag = "\"pruned-delete\"";
    server = std::make_unique<cucascade::test::loopback_range_server>(bytes, policy);
    query(con,
          "CREATE SECRET pruned_delete (TYPE SIRIUS_S3, SCOPE 's3://pruned-delete/', "
          "KEY_ID 'access', SECRET 'secret', REGION 'us-east-1', ENDPOINT " +
            duckdb::Value(server->endpoint()).ToSQLString() + ")");
    query(con,
          "UPDATE __ducklake_metadata_lake.ducklake_delete_file SET "
          "path='s3://pruned-delete/delete', path_is_relative=false WHERE end_snapshot IS NULL");
  }
  auto runtime = con.context->registered_state->Get<duckdb::SiriusContext>("sirius_state");
  REQUIRE(runtime);
  auto counters         = runtime->physical_counters();
  counters->track_units = true;
  std::atomic<uint64_t> decoded_rows{0};
  std::atomic<bool> reader_pushdown{false};
  counters->parquet_filter_for_testing = [&](bool pushed, int64_t rows) {
    if (pushed) reader_pushdown = true;
    decoded_rows += rows;
  };
  auto clear =
    std::shared_ptr<void>(nullptr, [&](void*) { counters->parquet_filter_for_testing = {}; });
  query(con, "SET enable_duckdb_fallback=false");
  for (size_t i = 0; i < statements.size(); ++i) {
    auto const& sql = statements[i];
    INFO(sql);
    query(con, "SET gpu_execution=true");
    for (auto const& statement :
         {sql, "CALL gpu_execution(" + duckdb::Value(sql).ToSQLString() + ")"}) {
      auto before      = runtime->get_transparent_execution_stats();
      auto rows_before = decoded_rows.load();
      CHECK(ducklake_result_bag(*query(con, statement)) == expected[i]);
      auto after = runtime->get_transparent_execution_stats();
      CHECK(after.fallbacks == before.fallbacks);
      CHECK(after.runtime_fallbacks == before.runtime_fallbacks);
      // Exactly the second physical group is decoded. Deletes must still use
      // original file positions, including a future delete at old snapshots.
      CHECK(decoded_rows.load() - rows_before == 122880);
      CHECK_FALSE(reader_pushdown.load());
    }
  }
  if (remote) {
    CHECK(server->head_count() > 0);
    CHECK(server->get_count() > 0);
  }
}

TEST_CASE_METHOD(snapshot_fixture,
                 "DuckLake snapshot visibility pruned row groups retain original delete positions",
                 "[ducklake_snapshot_visibility_gpu][isolated_context]")
{
  check_pruned_delete_positions(false);
}

TEST_CASE_METHOD(snapshot_fixture,
                 "DuckLake remote deletes retain original positions after row group pruning",
                 "[.][ducklake_remote_pruning][isolated_context]")
{
  check_pruned_delete_positions(true);
}

TEST_CASE_METHOD(
  snapshot_fixture,
  "DuckLake snapshot visibility prepared scans refresh views across local changes and commits",
  "[ducklake_snapshot_visibility_gpu][ducklake_snapshot_visibility_prepared][isolated_context]")
{
  auto vectors = GENERATE(false, true);
  INFO("deletion vectors: " << vectors);
  run("SET gpu_execution=false");
  attach_lake(con, disk.directory);
  run("CALL lake.set_option('data_inlining_row_limit', 0)");
  run(std::string("CALL lake.set_option('write_deletion_vectors', ") +
      (vectors ? "true)" : "false)"));
  run("CREATE TABLE lake.q(id BIGINT,value BIGINT)");
  run("INSERT INTO lake.q SELECT i,CASE WHEN i%7=0 THEN NULL ELSE i%7 END FROM range(128) r(i)");
  auto historical = run("SELECT max(snapshot_id) FROM ducklake_snapshots('lake')")
                      ->Collection()
                      .GetValue(0, 0)
                      .ToString();
  run("INSERT INTO lake.q SELECT i,i%7 FROM range(128,256) r(i)");
  REQUIRE(run("CALL ducklake_merge_adjacent_files('lake', 'q')")->RowCount() == 1);
  run("DELETE FROM lake.q WHERE id=1");
  run("DELETE FROM lake.q WHERE id=129");
  REQUIRE(run("SELECT count(*) FROM __ducklake_metadata_lake.ducklake_data_file "
              "WHERE end_snapshot IS NULL AND partial_max IS NOT NULL")
            ->Collection()
            .GetValue(0, 0)
            .GetValue<int64_t>() == 1);
  auto deletes = run(
    "SELECT format,partial_max FROM __ducklake_metadata_lake.ducklake_delete_file "
    "WHERE end_snapshot IS NULL");
  REQUIRE(deletes->RowCount() == 1);
  REQUIRE(deletes->Collection().GetValue(0, 0).ToString() == (vectors ? "puffin" : "parquet"));
  REQUIRE_FALSE(deletes->Collection().GetValue(1, 0).IsNull());
  run("CALL lake.set_option('data_inlining_row_limit',100)");
  run("INSERT INTO lake.q VALUES (300,30),(300,30),(301,NULL)");
  auto runtime = con.context->registered_state->Get<duckdb::SiriusContext>("sirius_state");
  REQUIRE(runtime);
  auto counters         = runtime->physical_counters();
  counters->track_units = true;
  auto hosts =
    runtime->get_memory_manager().get_memory_spaces_for_tier(cucascade::memory::Tier::HOST);
  REQUIRE(hosts.size() == 1);
  auto reserved = hosts[0]->get_total_reserved_memory();
  std::atomic<unsigned> decoded{0}, parquet_visits{0};
  counters->parquet_phase_for_testing = [&](std::string const&, bool footer) {
    ++parquet_visits;
    if (!footer) ++decoded;
  };
  auto clear =
    std::shared_ptr<void>(nullptr, [&](void*) { counters->parquet_phase_for_testing = {}; });
  std::array<std::string, 2> prefixes{
    "SELECT value,id FROM lake.q WHERE id>=",
    "SELECT value,id FROM lake.q AT (VERSION => " + historical + ") WHERE id>="};
  std::array<std::vector<std::string>, 2> initial;
  for (size_t i = 0; i < prefixes.size(); ++i)
    initial[i] = ducklake_result_bag(*run(prefixes[i] + "0"));
  run("SET gpu_execution=true");
  run("SET enable_duckdb_fallback=false");
  auto before_prepare         = runtime->get_transparent_execution_stats();
  auto imports_before_prepare = counters->arrow_rows_imported.load();
  auto current                = con.Prepare(prefixes[0] + "?");
  auto frozen                 = con.Prepare(prefixes[1] + "?");
  REQUIRE_FALSE(current->HasError());
  REQUIRE_FALSE(frozen->HasError());
  CHECK(parquet_visits.load() == 0);
  CHECK(counters->arrow_rows_imported.load() == imports_before_prepare);
  CHECK(runtime->get_transparent_execution_stats().executions == before_prepare.executions);
  CHECK(hosts[0]->get_total_reserved_memory() == reserved);

  auto verify = [&](std::string const& stage, size_t visible_rows) {
    INFO(stage);
    for (auto threshold : {int64_t{0}, int64_t{64}, int64_t{128}, int64_t{0}}) {
      INFO(threshold);
      for (size_t i = 0; i < prefixes.size(); ++i) {
        INFO((i == 0 ? "current view" : "fixed historical view"));
        // The CPU oracle shares the original session and transaction, including
        // uncommitted overlays. The historical selector excludes those overlays.
        run("SET gpu_execution=false");
        auto native   = run(prefixes[i] + std::to_string(threshold));
        auto expected = ducklake_result_bag(*native);
        if (threshold == 0) {
          CHECK(expected.size() == (i == 0 ? visible_rows : 128));
          if (i == 1) CHECK(expected == initial[1]);
        }
        run("SET gpu_execution=true");
        for (int repeat = 0; repeat < 2; ++repeat) {
          auto before  = runtime->get_transparent_execution_stats();
          auto reads   = decoded.load();
          auto imports = counters->arrow_rows_imported.load();
          auto result  = (i == 0 ? current : frozen)->Execute(threshold);
          REQUIRE(result);
          INFO((result->HasError() ? result->GetError() : "prepared execution completed"));
          REQUIRE_FALSE(result->HasError());
          CHECK(ducklake_result_bag(*result) == expected);
          CHECK(result->GetTypes() == native->GetTypes());
          CHECK(result->GetNames() == native->GetNames());
          auto after = runtime->get_transparent_execution_stats();
          CHECK(after.executions == before.executions + 1);
          CHECK(after.fallbacks == before.fallbacks);
          CHECK(after.runtime_fallbacks == before.runtime_fallbacks);
          if (!expected.empty()) CHECK(decoded.load() > reads);
          if (i == 0) CHECK(counters->arrow_rows_imported.load() > imports);
          result.reset();
          CHECK(hosts[0]->get_total_reserved_memory() == reserved);
        }
      }
    }
  };
  verify("initial committed mixed view", 257);
  run("SET gpu_execution=false");
  run("DELETE FROM lake.q WHERE id=3");
  verify("delete-only commit refreshes prepared visibility", 256);
  run("SET gpu_execution=false");
  run("BEGIN");
  run("INSERT INTO lake.q VALUES (350,35),(350,35),(351,NULL)");
  run("DELETE FROM lake.q WHERE id IN (2,130,300)");
  verify("read own inserts and deletes over snapshot-bearing files", 255);
  run("SET gpu_execution=false");
  run("ROLLBACK");
  verify("rollback restores committed visibility", 256);
  run("SET gpu_execution=false");
  run("BEGIN");
  run("INSERT INTO lake.q VALUES (350,35),(350,35),(351,NULL)");
  run("DELETE FROM lake.q WHERE id IN (4,301)");
  run("COMMIT");
  verify("commit refreshes prepared units without changing historical view", 257);
  // A later query remains usable while the old prepared plans are still alive.
  run("SET gpu_execution=false");
  auto final_expected = ducklake_result_bag(*run(prefixes[0] + "0"));
  REQUIRE(final_expected != initial[0]);
  run("SET gpu_execution=true");
  auto before_fresh = runtime->get_transparent_execution_stats();
  auto fresh_reads  = decoded.load();
  CHECK(ducklake_result_bag(*run(prefixes[0] + "0")) == final_expected);
  auto after_fresh = runtime->get_transparent_execution_stats();
  CHECK(after_fresh.executions == before_fresh.executions + 1);
  CHECK(after_fresh.fallbacks == before_fresh.fallbacks);
  CHECK(after_fresh.runtime_fallbacks == before_fresh.runtime_fallbacks);
  CHECK(decoded.load() > fresh_reads);
  current.reset();
  frozen.reset();
  CHECK(hosts[0]->get_total_reserved_memory() == reserved);
}

namespace {
void check_snapshot_result(duckdb::QueryResult& result,
                           duckdb::QueryResult const& native,
                           std::vector<std::string> const& expected)
{
  CHECK(ducklake_result_bag(result) == expected);
  CHECK(result.GetTypes() == native.GetTypes());
  CHECK(result.GetNames() == native.GetNames());
}

// Native A has deletes, so compaction merges only B/C. At the selected version,
// C and the later file deletes are invisible, while committed inline rows remain.
std::string snapshot_visibility_recovery_query(duckdb::Connection& con,
                                               bool vectors,
                                               uint64_t rows_per_file = 4096)
{
  auto run = [&](std::string const& sql) { return query(con, sql); };
  run("CALL lake.set_option('data_inlining_row_limit',0)");
  run(std::string("CALL lake.set_option('write_deletion_vectors',") +
      (vectors ? "true)" : "false)"));
  run("CREATE TABLE lake.q(id BIGINT,value BIGINT)");
  run("INSERT INTO lake.q SELECT i,i%7 FROM range(" + std::to_string(rows_per_file) + ") r(i)");
  run("DELETE FROM lake.q WHERE id=1");
  run("CALL lake.set_option('data_inlining_row_limit',100)");
  run("INSERT INTO lake.q VALUES (20000,5),(20000,5),(20001,NULL)");
  run("CALL lake.set_option('data_inlining_row_limit',0)");
  run("INSERT INTO lake.q SELECT i,i%7 FROM range(" + std::to_string(rows_per_file) + "," +
      std::to_string(2 * rows_per_file) + ") r(i)");
  auto version = run("SELECT max(snapshot_id) FROM ducklake_snapshots('lake')")
                   ->Collection()
                   .GetValue(0, 0)
                   .ToString();
  run("INSERT INTO lake.q SELECT i,i%7 FROM range(" + std::to_string(2 * rows_per_file) + "," +
      std::to_string(3 * rows_per_file) + ") r(i)");
  REQUIRE(run("CALL ducklake_merge_adjacent_files('lake', 'q')")->RowCount() == 1);
  run("DELETE FROM lake.q WHERE id=2");
  run("DELETE FROM lake.q WHERE id=" + std::to_string(rows_per_file + 2));
  REQUIRE(run("SELECT count(*) FROM __ducklake_metadata_lake.ducklake_data_file "
              "WHERE end_snapshot IS NULL AND partial_max IS NOT NULL")
            ->Collection()
            .GetValue(0, 0)
            .GetValue<int64_t>() == 1);
  auto deletes = run(
    "SELECT format,partial_max FROM __ducklake_metadata_lake.ducklake_delete_file "
    "WHERE end_snapshot IS NULL");
  REQUIRE(deletes->RowCount() == 2);
  bool snapshot_delete = false;
  for (duckdb::idx_t i = 0; i < deletes->RowCount(); ++i) {
    REQUIRE(deletes->Collection().GetValue(0, i).ToString() == (vectors ? "puffin" : "parquet"));
    snapshot_delete |= !deletes->Collection().GetValue(1, i).IsNull();
  }
  REQUIRE(snapshot_delete);
  return "SELECT value,id FROM lake.q AT (VERSION => " + version + ") WHERE id>=";
}
}  // namespace

TEST_CASE_METHOD(
  snapshot_fixture,
  "DuckLake snapshot visibility failed attempts release visibility before replay and prepared "
  "reuse",
  "[ducklake_snapshot_visibility_gpu][ducklake_snapshot_visibility_recovery][isolated_context]")
{
  auto vectors        = GENERATE(false, true);
  auto explicit_entry = GENERATE(false, true);
  auto fallback       = GENERATE(false, true);
  CAPTURE(vectors, explicit_entry, fallback);
  run("SET gpu_execution=false");
  attach_lake(con, disk.directory);
  auto prefix = snapshot_visibility_recovery_query(con, vectors);
  auto native = run(prefix + "0");
  std::array<std::vector<std::string>, 2> expected{ducklake_result_bag(*native),
                                                   ducklake_result_bag(*run(prefix + "1024"))};
  REQUIRE(expected[0].size() == 8194);
  REQUIRE(expected[1].size() == 7171);
  auto runtime = con.context->registered_state->Get<duckdb::SiriusContext>("sirius_state");
  REQUIRE(runtime);
  auto counters         = runtime->physical_counters();
  counters->track_units = true;
  auto hosts =
    runtime->get_memory_manager().get_memory_spaces_for_tier(cucascade::memory::Tier::HOST);
  REQUIRE(hosts.size() == 1);
  auto reserved = hosts[0]->get_total_reserved_memory();
  std::atomic<unsigned> decoded{0}, replays{0};
  std::atomic<bool> inject{false};
  std::atomic<uint64_t> reserved_at_replay{UINT64_MAX};
  counters->parquet_filter_for_testing = [&](bool, int64_t rows) {
    if (rows > 0 && ++decoded == 2 && inject.load())
      throw sirius::transparent::classified_execution_error(
        sirius::transparent::late_failure_cause::reader_io,
        "injected snapshot visibility post-decode failure");
  };
  runtime->cpu_replay_hook_for_testing = [&] {
    ++replays;
    reserved_at_replay = hosts[0]->get_total_reserved_memory();
  };
  auto clear = std::shared_ptr<void>(nullptr, [&](void*) {
    counters->parquet_filter_for_testing = {};
    runtime->cpu_replay_hook_for_testing = {};
  });
  run("SET gpu_execution=true");
  run("SET scan_task_batch_size=1");
  run("SET sirius_test_sync_cpu_replay=true");
  run("SET enable_duckdb_fallback=false");
  auto prepared = con.Prepare(prefix + "?");
  REQUIRE_FALSE(prepared->HasError());
  auto execute = [&](int64_t parameter) {
    if (explicit_entry)
      return con.Query("CALL gpu_execution(" +
                       duckdb::Value(prefix + std::to_string(parameter)).ToSQLString() + ")");
    return prepared->Execute(parameter);
  };
  CHECK(ducklake_result_bag(*execute(0)) == expected[0]);
  REQUIRE(decoded >= 2);
  CHECK(hosts[0]->get_total_reserved_memory() == reserved);
  for (auto parameter : {int64_t{0}, int64_t{1024}, int64_t{0}}) {
    INFO(parameter);
    auto const& oracle = expected[parameter == 0 ? 0 : 1];
    run(std::string("SET enable_duckdb_fallback=") + (fallback ? "true" : "false"));
    decoded            = 0;
    inject             = true;
    reserved_at_replay = UINT64_MAX;
    auto replay_before = replays.load();
    auto before        = runtime->get_transparent_execution_stats();
    auto failed        = execute(parameter);
    REQUIRE(decoded >= 2);
    INFO((failed->HasError() ? failed->GetError() : "replay completed"));
    if (fallback) {
      REQUIRE_FALSE(failed->HasError());
      check_snapshot_result(*failed, *native, oracle);
      CHECK(reserved_at_replay.load() == reserved);
    } else {
      REQUIRE(failed->HasError());
      CHECK(failed->GetError().find("injected snapshot visibility post-decode failure") !=
            std::string::npos);
    }
    CHECK(replays.load() == replay_before + (fallback ? 1 : 0));
    CHECK(runtime->get_transparent_execution_stats().fallbacks == before.fallbacks);
    CHECK(runtime->get_transparent_execution_stats().runtime_fallbacks ==
          before.runtime_fallbacks + (fallback ? 1 : 0));
    failed.reset();
    CHECK(hosts[0]->get_total_reserved_memory() == reserved);
    inject  = false;
    decoded = 0;
    run("SET enable_duckdb_fallback=false");
    auto imports      = counters->arrow_rows_imported.load();
    auto clean_before = runtime->get_transparent_execution_stats();
    auto clean        = execute(parameter);
    REQUIRE_FALSE(clean->HasError());
    check_snapshot_result(*clean, *native, oracle);
    CHECK(decoded >= 2);
    CHECK(counters->arrow_rows_imported.load() > imports);
    auto clean_after = runtime->get_transparent_execution_stats();
    CHECK(clean_after.fallbacks == clean_before.fallbacks);
    CHECK(clean_after.runtime_fallbacks == clean_before.runtime_fallbacks);
    if (!explicit_entry) CHECK(clean_after.executions == clean_before.executions + 1);
    CHECK(replays.load() == replay_before + (fallback ? 1 : 0));
    clean.reset();
    CHECK(hosts[0]->get_total_reserved_memory() == reserved);
  }
  prepared.reset();
  CHECK(hosts[0]->get_total_reserved_memory() == reserved);
}

TEST_CASE_METHOD(
  snapshot_fixture,
  "DuckLake snapshot visibility cancellation drains reads and allows a fresh query without replay",
  "[ducklake_snapshot_visibility_gpu][ducklake_snapshot_visibility_cancel][isolated_context]")
{
  auto vectors        = GENERATE(false, true);
  auto explicit_entry = GENERATE(false, true);
  auto footer         = GENERATE(false, true);
  CAPTURE(vectors, explicit_entry, footer);
  run("SET gpu_execution=false");
  attach_lake(con, disk.directory);
  auto sql      = snapshot_visibility_recovery_query(con, vectors) + "0";
  auto native   = run(sql);
  auto expected = ducklake_result_bag(*native);
  REQUIRE(expected.size() == 8194);
  auto runtime = con.context->registered_state->Get<duckdb::SiriusContext>("sirius_state");
  REQUIRE(runtime);
  auto counters         = runtime->physical_counters();
  counters->track_units = true;
  auto hosts =
    runtime->get_memory_manager().get_memory_spaces_for_tier(cucascade::memory::Tier::HOST);
  REQUIRE(hosts.size() == 1);
  auto reserved = hosts[0]->get_total_reserved_memory();
  std::promise<void> arrived, release;
  auto at_read = arrived.get_future();
  auto resume  = release.get_future().share();
  std::atomic<bool> first{true};
  std::atomic<unsigned> replays{0}, decoded{0};
  counters->parquet_phase_for_testing = [&](std::string const&, bool at_footer) {
    if (at_footer == footer && first.exchange(false)) {
      arrived.set_value();
      if (resume.wait_for(std::chrono::seconds(15)) != std::future_status::ready)
        throw std::runtime_error("snapshot visibility read rendezvous timed out");
    }
  };
  counters->parquet_filter_for_testing = [&](bool, int64_t rows) {
    if (rows > 0) ++decoded;
  };
  runtime->cpu_replay_hook_for_testing = [&] { ++replays; };
  auto clear                           = std::shared_ptr<void>(nullptr, [&](void*) {
    counters->parquet_phase_for_testing  = {};
    counters->parquet_filter_for_testing = {};
    runtime->cpu_replay_hook_for_testing = {};
  });
  run("SET gpu_execution=true");
  run("SET scan_task_batch_size=1");
  run("SET enable_duckdb_fallback=true");
  run("SET sirius_test_sync_cpu_replay=true");
  auto statement =
    explicit_entry ? "CALL gpu_execution(" + duckdb::Value(sql).ToSQLString() + ")" : sql;
  auto before   = runtime->get_transparent_execution_stats();
  auto active   = std::async(std::launch::async, [&] { return con.Query(statement); });
  bool released = false;
  auto unblock  = std::shared_ptr<void>(nullptr, [&](void*) {
    if (!released) release.set_value();
  });
  REQUIRE(at_read.wait_for(std::chrono::seconds(15)) == std::future_status::ready);
  con.Interrupt();
  // Cancellation must wait for the blocked reader to leave before returning.
  CHECK(active.wait_for(std::chrono::milliseconds(100)) == std::future_status::timeout);
  release.set_value();
  released = true;
  REQUIRE(active.wait_for(std::chrono::seconds(15)) == std::future_status::ready);
  auto cancelled = active.get();
  REQUIRE(cancelled->HasError());
  INFO(cancelled->GetError());
  CHECK(cancelled->GetErrorType() == duckdb::ExceptionType::INTERRUPT);
  CHECK(replays.load() == 0);
  CHECK(runtime->get_transparent_execution_stats().runtime_fallbacks == before.runtime_fallbacks);
  cancelled.reset();
  CHECK(hosts[0]->get_total_reserved_memory() == reserved);
  counters->parquet_phase_for_testing = {};
  decoded                             = 0;
  run("SET sirius_test_sync_cpu_replay=false");
  run("SET enable_duckdb_fallback=false");
  auto imports      = counters->arrow_rows_imported.load();
  auto fresh_before = runtime->get_transparent_execution_stats();
  auto fresh        = run(statement);
  check_snapshot_result(*fresh, *native, expected);
  CHECK(decoded >= 2);
  CHECK(counters->arrow_rows_imported.load() > imports);
  auto fresh_after = runtime->get_transparent_execution_stats();
  CHECK(fresh_after.fallbacks == fresh_before.fallbacks);
  CHECK(fresh_after.runtime_fallbacks == fresh_before.runtime_fallbacks);
  CHECK(replays.load() == 0);
  fresh.reset();
  CHECK(hosts[0]->get_total_reserved_memory() == reserved);
}

namespace {
class snapshot_visibility_puffin_source final : public sirius::op::scan::puffin_byte_source {
 public:
  explicit snapshot_visibility_puffin_source(std::string const& path)
  {
    auto length = std::filesystem::file_size(path);
    REQUIRE(length > 0);
    REQUIRE(length <= 1024 * 1024);
    bytes_.resize(length);
    std::ifstream input(path, std::ios::binary);
    input.read(reinterpret_cast<char*>(bytes_.data()), bytes_.size());
    REQUIRE(input.good());
  }
  bool seek(std::streamoff offset) override
  {
    if (offset < 0 || static_cast<size_t>(offset) > bytes_.size()) return false;
    cursor_ = offset;
    return true;
  }
  bool read(std::span<std::byte> out) override
  {
    if (out.size() > bytes_.size() - cursor_) return false;
    std::copy_n(bytes_.data() + cursor_, out.size(), out.data());
    cursor_ += out.size();
    return true;
  }
  std::streamoff size() const override { return bytes_.size(); }

 private:
  std::vector<std::byte> bytes_;
  size_t cursor_ = 0;
};

std::vector<std::pair<uint64_t, int64_t>> snapshot_visibility_dv_crc_offsets(
  std::string const& path)
{
  namespace sm = sirius::scan_manager;
  snapshot_visibility_puffin_source source(path);
  sm::test::test_reservation_provider provider;
  provider.grant_bytes = 8 * 1024 * 1024;
  sm::preparation_ledger ledger(provider);
  sm::unit_envelope unit{{}, 4 * 1024 * 1024};
  std::array scans{sm::scan_envelope{{unit}, {}, true}};
  REQUIRE(ledger.admit(scans, sm::memory_space_id(cucascade::memory::Tier::HOST, 0)).deferred);
  auto permit = ledger.try_acquire_unit({1, 1}, unit.retained);
  REQUIRE(permit);
  auto allocator = ledger.allocator(permit);
  auto metadata  = sirius::op::scan::read_puffin_metadata(source, allocator);
  std::vector<std::pair<uint64_t, int64_t>> offsets;
  for (size_t i = 0; i < metadata->blob_count(); ++i) {
    auto blob = metadata->blob(i);
    if (blob.type != "deletion-vector-v1") continue;
    auto snapshot = metadata->property(i, "ducklake-snapshot-id");
    REQUIRE(snapshot);
    size_t consumed = 0;
    auto version    = std::stoull(std::string(*snapshot), &consumed);
    REQUIRE(consumed == snapshot->size());
    REQUIRE(blob.offset >= 4);
    REQUIRE(blob.length >= 12);
    REQUIRE(blob.offset <= source.size());
    REQUIRE(blob.length <= source.size() - blob.offset);
    // deletion-vector-v1 ends in a four-byte CRC. Alter only its final byte;
    // footer, bitmap, snapshot properties and catalog sizes remain native.
    offsets.emplace_back(version, blob.offset + blob.length - 1);
  }
  REQUIRE(offsets.size() >= 2);
  return offsets;
}

class snapshot_visibility_crc_mutation {
 public:
  snapshot_visibility_crc_mutation(std::string path, int64_t offset)
    : path_(std::move(path)),
      offset_(offset),
      original_time_(std::filesystem::last_write_time(path_))
  {
    std::ifstream input(path_, std::ios::binary);
    input.seekg(offset_);
    input.read(&original_, 1);
    if (!input) throw std::runtime_error("could not read native DV CRC byte");
  }
  snapshot_visibility_crc_mutation(snapshot_visibility_crc_mutation const&)            = delete;
  snapshot_visibility_crc_mutation& operator=(snapshot_visibility_crc_mutation const&) = delete;
  ~snapshot_visibility_crc_mutation()
  {
    if (changed_) {
      try {
        restore();
      } catch (...) { /* The temporary fixture is removed on unwind. */
      }
    }
  }
  void corrupt()
  {
    changed_ = true;
    write(static_cast<char>(static_cast<unsigned char>(original_) ^ 1), 2);
  }
  void restore()
  {
    write(original_, 4);
    changed_ = false;
  }

 private:
  void write(char value, int seconds)
  {
    std::fstream output(path_, std::ios::binary | std::ios::in | std::ios::out);
    output.seekp(offset_);
    output.write(&value, 1);
    output.flush();
    if (!output) throw std::runtime_error("could not write native DV CRC byte");
    output.close();
    // Make cache identity changes deterministic even on coarse timestamp clocks.
    std::filesystem::last_write_time(path_, original_time_ + std::chrono::seconds(seconds));
  }
  std::string path_;
  int64_t offset_;
  std::filesystem::file_time_type original_time_;
  char original_{};
  bool changed_ = false;
};
}  // namespace

TEST_CASE_METHOD(
  snapshot_fixture,
  "DuckLake snapshot visibility validates corrupt future DV blobs across repeated reads and replay",
  "[ducklake_snapshot_visibility_gpu][ducklake_snapshot_visibility_corrupt_dv][isolated_context]")
{
  auto future         = GENERATE(false, true);
  auto warm           = GENERATE(false, true);
  auto explicit_entry = GENERATE(false, true);
  auto fallback       = GENERATE(false, true);
  CAPTURE(future, warm, explicit_entry, fallback);
  run("SET gpu_execution=false");
  attach_lake(con, disk.directory);
  run("CALL lake.set_option('data_inlining_row_limit',0)");
  run("CALL lake.set_option('write_deletion_vectors',true)");
  run("CREATE TABLE lake.q AS SELECT i::BIGINT id FROM range(128) r(i)");
  run("DELETE FROM lake.q WHERE id=1");
  auto snapshot = run("SELECT max(snapshot_id) FROM ducklake_snapshots('lake')")
                    ->Collection()
                    .GetValue(0, 0)
                    .GetValue<uint64_t>();
  run("DELETE FROM lake.q WHERE id=2");
  auto sql      = "SELECT id FROM lake.q AT (VERSION => " + std::to_string(snapshot) + ")";
  auto native   = run(sql);
  auto expected = ducklake_result_bag(*native);
  REQUIRE(expected.size() == 127);
  auto files = run("SELECT delete_file FROM ducklake_list_files('lake','q', snapshot_version => " +
                   std::to_string(snapshot) + ")");
  REQUIRE(files->RowCount() == 1);
  auto path     = files->Collection().GetValue(0, 0).ToString();
  auto offsets  = snapshot_visibility_dv_crc_offsets(path);
  auto selected = std::find_if(offsets.begin(), offsets.end(), [&](auto const& entry) {
    return future ? entry.first > snapshot : entry.first <= snapshot;
  });
  REQUIRE(selected != offsets.end());
  if (future)
    REQUIRE(selected != offsets.begin());
  else
    REQUIRE(selected->first == snapshot);
  snapshot_visibility_crc_mutation mutation(path, selected->second);
  auto runtime = con.context->registered_state->Get<duckdb::SiriusContext>("sirius_state");
  REQUIRE(runtime);
  auto counters         = runtime->physical_counters();
  counters->track_units = true;
  auto hosts =
    runtime->get_memory_manager().get_memory_spaces_for_tier(cucascade::memory::Tier::HOST);
  REQUIRE(hosts.size() == 1);
  auto reserved = hosts[0]->get_total_reserved_memory();
  std::atomic<unsigned> decoded{0}, replays{0};
  std::atomic<uint64_t> reserved_at_replay{UINT64_MAX};
  counters->parquet_filter_for_testing = [&](bool, uint64_t rows) {
    if (rows) ++decoded;
  };
  runtime->cpu_replay_hook_for_testing = [&] {
    ++replays;
    reserved_at_replay = hosts[0]->get_total_reserved_memory();
  };
  auto clear = std::shared_ptr<void>(nullptr, [&](void*) {
    counters->parquet_filter_for_testing = {};
    runtime->cpu_replay_hook_for_testing = {};
  });
  run("SET gpu_execution=true");
  run("SET enable_duckdb_fallback=false");
  run("SET sirius_test_sync_cpu_replay=true");
  auto prepared = con.Prepare(sql);
  REQUIRE_FALSE(prepared->HasError());
  auto execute = [&] {
    if (explicit_entry)
      return con.Query("CALL gpu_execution(" + duckdb::Value(sql).ToSQLString() + ")");
    return prepared->Execute();
  };
  if (warm) {
    for (int i = 0; i < 2; ++i) {
      auto reads = decoded.load();
      CHECK(ducklake_result_bag(*execute()) == expected);
      CHECK(decoded.load() > reads);
      CHECK(hosts[0]->get_total_reserved_memory() == reserved);
    }
  }
  if (!warm) CHECK(decoded.load() == 0);
  mutation.corrupt();
  run("SET gpu_execution=false");
  auto native_error = con.Query(sql);
  REQUIRE(native_error->HasError());
  REQUIRE(native_error->GetError().find("checksum") != std::string::npos);
  run("SET gpu_execution=true");
  run(std::string("SET enable_duckdb_fallback=") + (fallback ? "true" : "false"));
  for (int repeat = 0; repeat < 2; ++repeat) {
    INFO(repeat);
    decoded            = 0;
    auto replay_before = replays.load();
    auto before        = runtime->get_transparent_execution_stats();
    auto result        = execute();
    REQUIRE(result->HasError());
    INFO(result->GetError());
    CHECK(result->GetError().find(fallback ? "checksum" : "CRC-32 mismatch") != std::string::npos);
    if (fallback) {
      CHECK(result->GetErrorType() == native_error->GetErrorType());
      CHECK(reserved_at_replay.load() == reserved);
    }
    CHECK(decoded.load() == 0);
    CHECK(replays.load() == replay_before + (fallback ? 1 : 0));
    auto after = runtime->get_transparent_execution_stats();
    // This counter records an attempted GPU execution, before its result exists.
    if (!explicit_entry) CHECK(after.executions == before.executions + 1);
    CHECK(after.fallbacks == before.fallbacks);
    CHECK(after.runtime_fallbacks == before.runtime_fallbacks + (fallback ? 1 : 0));
    result.reset();
    CHECK(hosts[0]->get_total_reserved_memory() == reserved);
  }
  mutation.restore();
  run("SET gpu_execution=false");
  CHECK(ducklake_result_bag(*run(sql)) == expected);
  run("SET gpu_execution=true");
  run("SET enable_duckdb_fallback=false");
  decoded            = 0;
  auto replay_before = replays.load();
  auto before        = runtime->get_transparent_execution_stats();
  auto restored      = execute();
  REQUIRE_FALSE(restored->HasError());
  check_snapshot_result(*restored, *native, expected);
  CHECK(decoded.load() > 0);
  CHECK(replays.load() == replay_before);
  auto after = runtime->get_transparent_execution_stats();
  CHECK(after.fallbacks == before.fallbacks);
  CHECK(after.runtime_fallbacks == before.runtime_fallbacks);
  restored.reset();
  prepared.reset();
  CHECK(hosts[0]->get_total_reserved_memory() == reserved);
}

TEST_CASE_METHOD(
  snapshot_fixture,
  "DuckLake snapshot visibility concurrent commits refresh visibility while reusing physical "
  "metadata",
  "[ducklake_snapshot_visibility_gpu][ducklake_snapshot_visibility_concurrent_cache][isolated_"
  "context]")
{
  auto vectors        = GENERATE(false, true);
  auto explicit_entry = GENERATE(false, true);
  auto after_decode   = GENERATE(false, true);
  CAPTURE(vectors, explicit_entry, after_decode);
  run("SET gpu_execution=false");
  attach_lake(con, disk.directory);
  auto historical = snapshot_visibility_recovery_query(con, vectors);
  run("CALL lake.set_option('data_inlining_row_limit',100)");
  std::array<std::string, 2> prefixes{"SELECT value,id FROM lake.q WHERE id>=", historical};
  auto native = run(prefixes[0] + "0");
  std::array<std::vector<std::string>, 2> expected{ducklake_result_bag(*native),
                                                   ducklake_result_bag(*run(prefixes[1] + "0"))};
  REQUIRE(expected[0].size() == 12288);
  REQUIRE(expected[1].size() == 8194);
  auto files = run("SELECT data_file FROM ducklake_list_files('lake','q')");
  REQUIRE(files->RowCount() == 2);
  std::set<std::string> paths;
  struct file_identity {
    std::string path;
    uintmax_t size;
    std::filesystem::file_time_type modified;
    std::weak_ptr<cucascade::io::io_object_metadata> metadata;
  };
  std::vector<file_identity> identities;
  for (duckdb::idx_t i = 0; i < files->RowCount(); ++i) {
    auto path = files->Collection().GetValue(0, i).ToString();
    paths.insert(path);
    identities.push_back(
      {path, std::filesystem::file_size(path), std::filesystem::last_write_time(path), {}});
  }
  auto expected_files = ducklake_result_bag(*files);
  auto writer         = env.make_connection();
  query(writer, "SET gpu_execution=false");
  auto runtime = con.context->registered_state->Get<duckdb::SiriusContext>("sirius_state");
  REQUIRE(runtime);
  auto counters         = runtime->physical_counters();
  counters->track_units = true;
  auto hosts =
    runtime->get_memory_manager().get_memory_spaces_for_tier(cucascade::memory::Tier::HOST);
  REQUIRE(hosts.size() == 1);
  auto reserved = hosts[0]->get_total_reserved_memory();
  std::atomic<unsigned> decoded{0};
  std::mutex metadata_mutex;
  std::set<std::string> hits;
  counters->parquet_metadata_for_testing = [&](std::string const& path, bool hit) {
    std::lock_guard lock(metadata_mutex);
    if (hit) hits.insert(path);
  };
  counters->parquet_filter_for_testing = [&](bool, uint64_t rows) {
    if (rows) ++decoded;
  };
  auto clear = std::shared_ptr<void>(nullptr, [&](void*) {
    counters->parquet_metadata_for_testing = {};
    counters->parquet_filter_for_testing   = {};
    counters->after_certify_for_testing    = {};
  });
  run("SET gpu_execution=true");
  run("SET enable_duckdb_fallback=false");
  run("SET scan_task_batch_size=1");
  auto current = con.Prepare(prefixes[0] + "?");
  auto frozen  = con.Prepare(prefixes[1] + "?");
  REQUIRE_FALSE(current->HasError());
  REQUIRE_FALSE(frozen->HasError());
  auto execute = [&](size_t view) {
    if (explicit_entry)
      return con.Query("CALL gpu_execution(" + duckdb::Value(prefixes[view] + "0").ToSQLString() +
                       ")");
    return (view == 0 ? current : frozen)->Execute(int64_t{0});
  };
  auto verify = [&](size_t view, bool require_hits, auto&& action) {
    auto before  = runtime->get_transparent_execution_stats();
    auto reads   = decoded.load();
    auto imports = counters->arrow_rows_imported.load();
    {
      std::lock_guard lock(metadata_mutex);
      hits.clear();
    }
    auto result = action();
    REQUIRE_FALSE(result->HasError());
    check_snapshot_result(*result, *native, expected[view]);
    CHECK(decoded.load() >= reads + 2);
    CHECK(counters->arrow_rows_imported.load() > imports);
    if (require_hits) {
      std::lock_guard lock(metadata_mutex);
      CHECK(hits == paths);
    }
    auto after = runtime->get_transparent_execution_stats();
    if (!explicit_entry) CHECK(after.executions == before.executions + 1);
    CHECK(after.fallbacks == before.fallbacks);
    CHECK(after.runtime_fallbacks == before.runtime_fallbacks);
    result.reset();
    CHECK(hosts[0]->get_total_reserved_memory() == reserved);
    // A scan-time hit alone could come from this statement's certification.
    // Weak references also prove cross-statement reuse without extending ownership.
    for (auto& identity : identities) {
      auto source = runtime->get_scan_manager().create_datasource(identity.path);
      REQUIRE(source);
      auto metadata = source->metadata();
      REQUIRE(metadata);
      if (require_hits) {
        auto previous = identity.metadata.lock();
        REQUIRE(previous);
        CHECK(metadata == previous);
      } else {
        identity.metadata = metadata;
      }
    }
  };
  verify(0, false, [&] { return execute(0); });
  verify(1, true, [&] { return execute(1); });
  for (size_t view = 0; view < prefixes.size(); ++view) {
    INFO(view);
    verify(view, true, [&] {
      query_pause pause;
      std::atomic<bool> first{true};
      auto rendezvous = [&] {
        if (first.exchange(false)) { pause.arrive_and_wait(std::chrono::seconds(15)); }
      };
      if (after_decode) {
        counters->parquet_filter_for_testing = [&](bool, uint64_t rows) {
          if (rows) {
            ++decoded;
            rendezvous();
          }
        };
      } else {
        counters->after_certify_for_testing = rendezvous;
      }
      // Join the active query before resetting hooks or destroying their captures,
      // even when a writer or an assertion fails at the barrier.
      auto reset   = std::shared_ptr<void>(nullptr, [&](void*) {
        counters->after_certify_for_testing  = {};
        counters->parquet_filter_for_testing = [&](bool, uint64_t rows) {
          if (rows) ++decoded;
        };
      });
      auto active  = std::async(std::launch::async, [&] { return execute(view); });
      auto unblock = std::shared_ptr<void>(nullptr, [&](void*) { pause.release(); });
      REQUIRE(pause.wait_until_arrived());
      query(writer, "BEGIN");
      query(writer,
            "DELETE FROM lake.q WHERE id IN (" + std::to_string(3 + view) + "," +
              std::to_string(4099 + view) + ")");
      auto id = std::to_string(30000 + view);
      query(writer, "INSERT INTO lake.q VALUES (" + id + ",5),(" + id + ",5),(" + id + ",NULL)");
      query(writer, "COMMIT");
      CHECK(active.wait_for(std::chrono::milliseconds(100)) == std::future_status::timeout);
      pause.release();
      REQUIRE(active.wait_for(std::chrono::seconds(15)) == std::future_status::ready);
      return active.get();
    });
    // A new statement sees the commit, but the frozen selector still excludes it.
    run("SET gpu_execution=false");
    auto next = ducklake_result_bag(*run(prefixes[0] + "0"));
    CHECK(next != expected[0]);
    CHECK(next.size() == expected[0].size() + 1);
    expected[0] = std::move(next);
    CHECK(ducklake_result_bag(*run(prefixes[1] + "0")) == expected[1]);
    auto unchanged_files = run("SELECT data_file FROM ducklake_list_files('lake','q')");
    CHECK(ducklake_result_bag(*unchanged_files) == expected_files);
    REQUIRE(identities.size() == 2);
    for (auto const& identity : identities) {
      CHECK(std::filesystem::file_size(identity.path) == identity.size);
      CHECK(std::filesystem::last_write_time(identity.path) == identity.modified);
    }
    run("SET gpu_execution=true");
    verify(0, true, [&] { return execute(0); });
    verify(1, true, [&] { return execute(1); });
  }
  current.reset();
  frozen.reset();
  CHECK(hosts[0]->get_total_reserved_memory() == reserved);
}

TEST_CASE_METHOD(
  ducklake_fixture,
  "DuckLake historical deletes exclude newer snapshots during local deletion",
  "[ducklake_snapshot_visibility_gpu][ducklake_snapshot_delete_locality][isolated_context]")
{
  if (!sirius::ducklake::bridge_available()) SKIP("DuckLake private binding bridge is disabled");
  auto vectors        = GENERATE(false, true);
  auto explicit_entry = GENERATE(false, true);
  CAPTURE(vectors, explicit_entry);
  run("SET gpu_execution=false");
  attach_lake(con, disk.directory);
  run("CALL lake.set_option('data_inlining_row_limit',10)");
  run(std::string("CALL lake.set_option('write_deletion_vectors',") +
      (vectors ? "true)" : "false)"));
  run("CREATE TABLE lake.t AS SELECT i::BIGINT id FROM range(100) r(i)");
  run("DELETE FROM lake.t WHERE id=0");
  auto snapshot = run(
    "SELECT snapshot_id, snapshot_time FROM ducklake_snapshots('lake') "
    "ORDER BY snapshot_id DESC LIMIT 1");
  auto version   = snapshot->Collection().GetValue(0, 0).ToString();
  auto timestamp = snapshot->Collection().GetValue(1, 0).ToSQLString();
  run("DELETE FROM lake.t WHERE id=2");
  run("CALL ducklake_flush_inlined_data('lake')");
  auto format = run(
    "SELECT format FROM __ducklake_metadata_lake.ducklake_delete_file "
    "WHERE end_snapshot IS NULL");
  REQUIRE(format->RowCount() == 1);
  REQUIRE(format->Collection().GetValue(0, 0).ToString() == (vectors ? "puffin" : "parquet"));
  run("BEGIN");
  run("DELETE FROM lake.t WHERE id>=50 AND id<80");
  auto runtime          = sirius::test::get_registered_sirius_context(con);
  auto counters         = runtime->physical_counters();
  counters->track_units = true;
  std::atomic<unsigned> decoded{0};
  counters->parquet_filter_for_testing = [&](bool, uint64_t rows) {
    if (rows) ++decoded;
  };
  auto clear =
    std::shared_ptr<void>(nullptr, [&](void*) { counters->parquet_filter_for_testing = {}; });
  auto check = [&](std::string const& suffix, int64_t count, int64_t sum) {
    auto sql = "SELECT count(*), sum(id)::BIGINT FROM lake.t" + suffix;
    INFO(sql);
    run("SET gpu_execution=false");
    auto native = run(sql);
    REQUIRE(native->Collection().GetValue(0, 0).GetValue<int64_t>() == count);
    REQUIRE(native->Collection().GetValue(1, 0).GetValue<int64_t>() == sum);
    auto reads = decoded.load();
    check_gpu_result(sql, ducklake_result_bag(*native), explicit_entry);
    CHECK(decoded.load() > reads);
  };
  for (int repeat = 0; repeat < 2; ++repeat) {
    check("", 68, 3013);
    check(" AT (VERSION => " + version + ")", 99, 4950);
    check(" AT (TIMESTAMP => " + timestamp + ")", 99, 4950);
  }
  run("SET gpu_execution=false");
  run("ROLLBACK");
  check("", 98, 4948);
  check(" AT (VERSION => " + version + ")", 99, 4950);
}

TEST_CASE_METHOD(
  snapshot_fixture,
  "DuckLake snapshot visibility combines historical partial visibility with local overlays and "
  "defaults",
  "[ducklake_snapshot_visibility_gpu][ducklake_snapshot_visibility_local_joint][isolated_context]")
{
  auto vectors        = GENERATE(false, true);
  auto explicit_entry = GENERATE(false, true);
  CAPTURE(vectors, explicit_entry);
  run("SET gpu_execution=false");
  attach_lake(con, disk.directory);
  auto historical = snapshot_visibility_recovery_query(con, vectors) + "0";
  run("ALTER TABLE lake.q ADD COLUMN extra BIGINT DEFAULT 7");
  run("CALL lake.set_option('data_inlining_row_limit',100)");
  auto runtime = con.context->registered_state->Get<duckdb::SiriusContext>("sirius_state");
  REQUIRE(runtime);
  auto counters         = runtime->physical_counters();
  counters->track_units = true;
  auto hosts =
    runtime->get_memory_manager().get_memory_spaces_for_tier(cucascade::memory::Tier::HOST);
  REQUIRE(hosts.size() == 1);
  auto reserved = hosts[0]->get_total_reserved_memory();
  std::atomic<unsigned> decoded{0};
  counters->parquet_filter_for_testing = [&](bool, uint64_t rows) {
    if (rows) ++decoded;
  };
  auto clear =
    std::shared_ptr<void>(nullptr, [&](void*) { counters->parquet_filter_for_testing = {}; });
  run("SET enable_duckdb_fallback=false");
  std::vector<std::string> queries{"SELECT id,value,extra FROM lake.q",
                                   historical,
                                   "SELECT q.id,q.value AS current_value,q.extra,h.value AS "
                                   "historical_value FROM lake.q q JOIN (" +
                                     historical + ") h ON q.id=h.id",
                                   "SELECT extra,COUNT(*),SUM(id) FROM lake.q GROUP BY extra"};
  auto original_historical = ducklake_result_bag(*run(historical));
  auto verify              = [&] {
    for (auto const& sql : queries) {
      INFO(sql);
      run("SET gpu_execution=false");
      auto native   = run(sql);
      auto expected = ducklake_result_bag(*native);
      REQUIRE_FALSE(expected.empty());
      if (sql == historical) CHECK(expected == original_historical);
      run("SET gpu_execution=true");
      auto before = runtime->get_transparent_execution_stats();
      auto reads  = decoded.load();
      auto result =
        run(explicit_entry ? "CALL gpu_execution(" + duckdb::Value(sql).ToSQLString() + ")" : sql);
      check_snapshot_result(*result, *native, expected);
      CHECK(decoded.load() >= reads + 2);
      auto after = runtime->get_transparent_execution_stats();
      if (!explicit_entry) CHECK(after.executions == before.executions + 1);
      CHECK(after.fallbacks == before.fallbacks);
      CHECK(after.runtime_fallbacks == before.runtime_fallbacks);
      result.reset();
      CHECK(hosts[0]->get_total_reserved_memory() == reserved);
    }
  };
  verify();
  for (bool commit : {false, true}) {
    INFO(commit);
    run("SET gpu_execution=false");
    run("BEGIN");
    // Delete from A, compacted B/C and the committed inline rows, then add local duplicates.
    run("DELETE FROM lake.q WHERE id IN (3,4099,20000)");
    run("INSERT INTO lake.q VALUES (35000,5,9),(35000,5,9),(35001,NULL,NULL)");
    CHECK(run("SELECT count(*) FROM lake.q WHERE id=20000")
            ->Collection()
            .GetValue(0, 0)
            .GetValue<int64_t>() == 0);
    verify();
    run("SET gpu_execution=false");
    run(commit ? "COMMIT" : "ROLLBACK");
    CHECK(run("SELECT count(*) FROM lake.q WHERE id=35000")
            ->Collection()
            .GetValue(0, 0)
            .GetValue<int64_t>() == (commit ? 2 : 0));
    verify();
  }
}

TEST_CASE_METHOD(
  snapshot_fixture,
  "DuckLake snapshot visibility active views survive concurrent inline flush",
  "[ducklake_snapshot_visibility_gpu][ducklake_snapshot_visibility_maintenance][isolated_context]")
{
  auto vectors        = GENERATE(false, true);
  auto explicit_entry = GENERATE(false, true);
  auto after_decode   = GENERATE(false, true);
  CAPTURE(vectors, explicit_entry, after_decode);
  run("SET gpu_execution=false");
  attach_lake(con, disk.directory);
  auto historical       = snapshot_visibility_recovery_query(con, vectors) + "0";
  std::string current   = "SELECT value,id FROM lake.q";
  auto native           = run(historical);
  auto expected         = ducklake_result_bag(*native);
  auto current_expected = ducklake_result_bag(*run(current));
  auto initial_files =
    ducklake_result_bag(*run("SELECT data_file FROM ducklake_list_files('lake','q')"));
  REQUIRE(initial_files.size() == 2);
  auto writer = env.make_connection();
  query(writer, "SET gpu_execution=false");
  auto runtime = con.context->registered_state->Get<duckdb::SiriusContext>("sirius_state");
  REQUIRE(runtime);
  auto counters         = runtime->physical_counters();
  counters->track_units = true;
  auto hosts =
    runtime->get_memory_manager().get_memory_spaces_for_tier(cucascade::memory::Tier::HOST);
  REQUIRE(hosts.size() == 1);
  auto reserved = hosts[0]->get_total_reserved_memory();
  std::atomic<unsigned> decoded{0};
  query_pause pause;
  std::atomic<bool> first{true};
  auto rendezvous = [&] {
    if (first.exchange(false)) { pause.arrive_and_wait(std::chrono::seconds(20)); }
  };
  counters->parquet_filter_for_testing = [&](bool, uint64_t rows) {
    if (rows) {
      ++decoded;
      if (after_decode) rendezvous();
    }
  };
  if (!after_decode) counters->after_certify_for_testing = rendezvous;
  auto clear = std::shared_ptr<void>(nullptr, [&](void*) {
    counters->parquet_filter_for_testing = {};
    counters->after_certify_for_testing  = {};
  });
  run("SET gpu_execution=true");
  run("SET enable_duckdb_fallback=false");
  run("SET scan_task_batch_size=1");
  auto before  = runtime->get_transparent_execution_stats();
  auto active  = std::async(std::launch::async, [&] {
    return con.Query(explicit_entry
                       ? "CALL gpu_execution(" + duckdb::Value(historical).ToSQLString() + ")"
                       : historical);
  });
  auto unblock = std::shared_ptr<void>(nullptr, [&](void*) { pause.release(); });
  REQUIRE(pause.wait_until_arrived());
  query(writer, "CALL ducklake_flush_inlined_data('lake')");
  auto flushed_files =
    ducklake_result_bag(*query(writer, "SELECT data_file FROM ducklake_list_files('lake','q')"));
  REQUIRE(flushed_files != initial_files);
  CHECK(active.wait_for(std::chrono::milliseconds(100)) == std::future_status::timeout);
  pause.release();
  REQUIRE(active.wait_for(std::chrono::seconds(15)) == std::future_status::ready);
  auto result = active.get();
  REQUIRE_FALSE(result->HasError());
  check_snapshot_result(*result, *native, expected);
  CHECK(decoded >= 2);
  result.reset();
  CHECK(hosts[0]->get_total_reserved_memory() == reserved);
  counters->after_certify_for_testing  = {};
  counters->parquet_filter_for_testing = [&](bool, uint64_t rows) {
    if (rows) ++decoded;
  };
  for (auto const& sql : {current, historical}) {
    run("SET gpu_execution=false");
    auto cpu = run(sql);
    auto bag = ducklake_result_bag(*cpu);
    CHECK(bag == (sql == current ? current_expected : expected));
    run("SET gpu_execution=true");
    auto reads = decoded.load();
    auto fresh =
      run(explicit_entry ? "CALL gpu_execution(" + duckdb::Value(sql).ToSQLString() + ")" : sql);
    check_snapshot_result(*fresh, *cpu, bag);
    CHECK(decoded > reads);
    fresh.reset();
    CHECK(hosts[0]->get_total_reserved_memory() == reserved);
  }
  auto after = runtime->get_transparent_execution_stats();
  if (!explicit_entry) CHECK(after.executions == before.executions + 3);
  CHECK(after.fallbacks == before.fallbacks);
  CHECK(after.runtime_fallbacks == before.runtime_fallbacks);
}

TEST_CASE_METHOD(
  snapshot_fixture,
  "DuckLake snapshot visibility drains speculative publication before replay",
  "[isolated_context][ducklake_snapshot_visibility_gpu][ducklake_snapshot_visibility_publication]")
{
  auto vectors = GENERATE(false, true);
  CAPTURE(vectors);
  auto explicit_entry = GENERATE(false, true);
  auto hold_batch     = GENERATE(false, true);
  auto fallback       = GENERATE(false, true);
  CAPTURE(explicit_entry, hold_batch, fallback);
  attach_lake(con, disk.directory);
  run("SET gpu_execution=false");
  auto historical = snapshot_visibility_recovery_query(con, vectors) + "0";
  std::string sql =
    "SELECT value,COUNT(*),SUM(id) FROM (" + historical + ") q WHERE id%3<>0 GROUP BY value";
  auto expected = ducklake_result_bag(*run(sql));
  run("SET gpu_execution=true");
  auto runtime = con.context->registered_state->Get<duckdb::SiriusContext>("sirius_state");
  REQUIRE(runtime);
  auto counters         = runtime->physical_counters();
  counters->track_units = true;
  auto hosts =
    runtime->get_memory_manager().get_memory_spaces_for_tier(cucascade::memory::Tier::HOST);
  REQUIRE(hosts.size() == 1);
  auto reserved = hosts[0]->get_total_reserved_memory();
  std::atomic<unsigned> footers{0}, replays{0}, decoded{0};
  counters->parquet_filter_for_testing = [&](bool, uint64_t rows) {
    if (rows) ++decoded;
  };
  std::promise<void> arrived, release;
  auto blocked                        = arrived.get_future();
  auto resume                         = release.get_future().share();
  counters->parquet_phase_for_testing = [&](std::string const&, bool footer) {
    if (footer && ++footers == 2) {
      arrived.set_value();
      if (resume.wait_for(std::chrono::seconds(20)) != std::future_status::ready)
        throw std::runtime_error("publication rendezvous timed out");
      throw sirius::transparent::classified_execution_error(
        sirius::transparent::late_failure_cause::reader_io,
        "injected later DuckLake footer failure");
    }
  };
  uint64_t reserved_at_replay          = UINT64_MAX;
  runtime->cpu_replay_hook_for_testing = [&] {
    ++replays;
    reserved_at_replay = hosts[0]->get_total_reserved_memory();
  };
  auto clear = std::shared_ptr<void>(nullptr, [&](void*) {
    counters->parquet_phase_for_testing  = {};
    runtime->cpu_replay_hook_for_testing = {};
    counters->parquet_filter_for_testing = {};
  });
  run("SET sirius_test_sync_cpu_replay=true");
  run("SET scan_task_batch_size=1");
  // Enable the shared publication observer without holding a real footer.
  run("SET sirius_test_hold_footer_index=" + std::to_string(UINT64_MAX));
  run(std::string("SET sirius_test_hold_published_batch=") + (hold_batch ? "true" : "false"));
  run(std::string("SET enable_duckdb_fallback=") + (fallback ? "true" : "false"));
  auto before    = runtime->get_transparent_execution_stats();
  auto active    = std::async(std::launch::async, [&] {
    return con.Query(explicit_entry ? "CALL gpu_execution(" + duckdb::Value(sql).ToSQLString() + ")"
                                    : sql);
  });
  bool released  = false;
  auto unblock   = std::shared_ptr<void>(nullptr, [&](void*) {
    if (!released) release.set_value();
  });
  bool reached   = blocked.wait_for(std::chrono::seconds(15)) == std::future_status::ready;
  bool published = reached && runtime->get_scan_manager().wait_for_publication_for_testing(
                                std::chrono::seconds(15));
  release.set_value();
  released    = true;
  auto result = active.get();
  REQUIRE(reached);
  REQUIRE(published);
  INFO((result->HasError() ? result->GetError() : "replayed successfully"));
  CHECK(result->HasError() == !fallback);
  if (fallback)
    CHECK(ducklake_result_bag(*result) == expected);
  else
    CHECK(result->GetError().find("injected later DuckLake footer failure") != std::string::npos);
  CHECK(replays == (fallback ? 1 : 0));
  if (fallback) CHECK(reserved_at_replay == reserved);
  result.reset();
  CHECK(hosts[0]->get_total_reserved_memory() == reserved);
  auto after = runtime->get_transparent_execution_stats();
  CHECK(after.runtime_fallbacks == before.runtime_fallbacks + (fallback ? 1 : 0));
  if (hold_batch) CHECK(after.discarded_speculative_work > before.discarded_speculative_work);
  counters->parquet_phase_for_testing = {};
  run("SET sirius_test_hold_published_batch=false");
  run("SET sirius_test_hold_footer_index=0");
  run("SET enable_duckdb_fallback=false");
  auto fresh_before = runtime->get_transparent_execution_stats();
  auto reads        = decoded.load();
  auto fresh =
    run(explicit_entry ? "CALL gpu_execution(" + duckdb::Value(sql).ToSQLString() + ")" : sql);
  CHECK(ducklake_result_bag(*fresh) == expected);
  CHECK(decoded >= reads + 2);
  CHECK(runtime->get_transparent_execution_stats().runtime_fallbacks ==
        fresh_before.runtime_fallbacks);
  fresh.reset();
  CHECK(hosts[0]->get_total_reserved_memory() == reserved);
}

TEST_CASE_METHOD(
  snapshot_fixture,
  "DuckLake snapshot visibility preserves owners and refuses replay after checkpoint cleanup "
  "failure",
  "[isolated_context][ducklake_snapshot_visibility_gpu][ducklake_snapshot_visibility_cleanup]")
{
  auto vectors = GENERATE(false, true);
  CAPTURE(vectors);
  auto explicit_entry = GENERATE(false, true);
  CAPTURE(explicit_entry);
  attach_lake(con, disk.directory);
  run("SET gpu_execution=false");
  auto historical = snapshot_visibility_recovery_query(con, vectors) + "0";
  run("ATTACH '" + (std::filesystem::path(disk.directory) / "keys.db").string() +
      "' AS native_store");
  run("CREATE TABLE native_store.keys AS SELECT range::BIGINT k FROM range(7)");
  run("CHECKPOINT native_store");
  run("SET gpu_execution=true");
  auto runtime = con.context->registered_state->Get<duckdb::SiriusContext>("sirius_state");
  REQUIRE(runtime);
  auto counters         = runtime->physical_counters();
  counters->track_units = true;
  std::atomic<unsigned> decoded{0}, replays{0};
  counters->parquet_filter_for_testing = [&](bool, uint64_t rows) {
    if (rows && ++decoded == 2)
      throw sirius::transparent::classified_execution_error(
        sirius::transparent::late_failure_cause::reader_io, "injected DuckLake decode failure");
  };
  runtime->cpu_replay_hook_for_testing = [&] { ++replays; };
  auto clear                           = std::shared_ptr<void>(nullptr, [&](void*) {
    counters->parquet_filter_for_testing = {};
    runtime->cpu_replay_hook_for_testing = {};
    // The injection happens after worker drains. Release retained test keys before teardown.
    runtime->get_scan_manager().reset_all();
  });
  run("SET enable_duckdb_fallback=true");
  run("SET sirius_test_sync_cpu_replay=true");
  run("SET sirius_test_inject_checkpoint_cleanup_failure=true");
  auto before = runtime->get_transparent_execution_stats();
  std::string sql =
    "SELECT q.id,b.k FROM (" + historical + ") q JOIN native_store.keys b ON q.value=b.k";
  auto result = con.Query(
    explicit_entry ? "CALL gpu_execution(" + duckdb::Value(sql).ToSQLString() + ")" : sql);
  REQUIRE(result->HasError());
  INFO(result->GetError());
  CHECK(decoded >= 2);
  CHECK(replays == 0);
  CHECK(runtime->get_transparent_execution_stats().runtime_fallbacks == before.runtime_fallbacks);
  CHECK(runtime->get_runtime_health() == duckdb::SiriusContext::runtime_health::UNAVAILABLE);
  CHECK(runtime->get_scan_manager().checkpoint_key_count() > 0);
}

TEST_CASE_METHOD(
  snapshot_fixture,
  "DuckLake snapshot visibility result fragments preserve joint visibility without replay",
  "[isolated_context][ducklake_snapshot_visibility_gpu][ducklake_snapshot_visibility_fragment]")
{
  auto vectors     = GENERATE(false, true);
  auto fail_decode = GENERATE(false, true);
  CAPTURE(vectors, fail_decode);
  query(con, "SET gpu_execution=false");
  attach_lake(con, disk.directory);
  auto sql      = snapshot_visibility_recovery_query(con, vectors) + "0";
  auto expected = ducklake_result_bag(*query(con, sql));
  auto runtime  = con.context->registered_state->Get<duckdb::SiriusContext>("sirius_state");
  REQUIRE(runtime);
  auto hosts =
    runtime->get_memory_manager().get_memory_spaces_for_tier(cucascade::memory::Tier::HOST);
  REQUIRE(hosts.size() == 1);
  auto reserved         = hosts[0]->get_total_reserved_memory();
  auto counters         = runtime->physical_counters();
  counters->track_units = true;
  con.context->registered_state->Insert(
    sirius::exec::stream_bind_catalog::kStateKey,
    duckdb::make_shared_ptr<sirius::exec::stream_bind_catalog>());
  std::atomic<unsigned> decoded{0};
  counters->parquet_filter_for_testing = [&](bool, uint64_t rows) {
    if (rows) {
      ++decoded;
      if (fail_decode && decoded == 2) throw std::runtime_error("fragment decode failure");
    }
  };
  auto observer          = duckdb::make_shared_ptr<provider_query_observer>();
  unsigned visits        = 0;
  bool inside            = false;
  observer->inspect_plan = [&](duckdb::ClientContext& context, duckdb::LogicalOperator& plan) {
    if (&context != con.context.get() || inside) return;
    inside     = true;
    auto leave = std::shared_ptr<void>(nullptr, [&](void*) { inside = false; });
    ++visits;
    sirius::exec::fragment_spec spec;
    spec.plan_source = [&](duckdb::ClientContext& current) {
      REQUIRE(&current == &context);
      auto prepared = duckdb::make_shared_ptr<duckdb::PreparedStatementData>(
        duckdb::StatementType::SELECT_STATEMENT);
      prepared->types = {duckdb::LogicalType::BIGINT, duckdb::LogicalType::BIGINT};
      prepared->names = {"value", "id"};
      return sirius::exec::bound_plan{plan.Copy(current), std::move(prepared)};
    };
    sirius::exec::streaming_fragment fragment(context, std::move(spec));
    fragment.build();
    if (fail_decode) {
      REQUIRE_THROWS_WITH(fragment.run(),
                          Catch::Matchers::ContainsSubstring("fragment decode failure"));
      REQUIRE_THROWS(fragment.take_result());
    } else {
      fragment.run();
      CHECK(ducklake_result_bag(*fragment.take_result()) == expected);
    }
  };
  auto clear = std::shared_ptr<void>(nullptr, [&](void*) {
    observer->inspect_plan               = {};
    counters->parquet_filter_for_testing = {};
  });
  duckdb::OptimizerExtension extension;
  extension.optimizer_info    = observer;
  extension.optimize_function = observe_provider_query;
  duckdb::OptimizerExtension::Register(duckdb::DBConfig::GetConfig(*con.context), extension);
  auto imported = counters->arrow_rows_imported.load();
  auto before   = runtime->get_transparent_execution_stats();
  CHECK(ducklake_result_bag(*query(con, sql)) == expected);
  CHECK(visits == 1);
  CHECK(decoded >= 2);
  if (!fail_decode) CHECK(counters->arrow_rows_imported.load() > imported);
  auto after = runtime->get_transparent_execution_stats();
  CHECK(after.fallbacks == before.fallbacks);
  CHECK(after.runtime_fallbacks == before.runtime_fallbacks);
  CHECK(hosts[0]->get_total_reserved_memory() == reserved);
}

TEST_CASE_METHOD(
  snapshot_fixture,
  "DuckLake snapshot visibility provider failures retain errors and cancellation drains without "
  "replay",
  "[isolated_context][ducklake_snapshot_visibility_gpu][ducklake_snapshot_visibility_provider]")
{
  auto vectors        = GENERATE(false, true);
  auto explicit_entry = GENERATE(false, true);
  auto inline_sql     = GENERATE(false, true);
  auto mode = GENERATE(0, 1, 2);  // cancel; provider failure without replay; failure with replay
  CAPTURE(vectors, explicit_entry, inline_sql, mode);
  run("SET gpu_execution=false");
  attach_lake(con, disk.directory);
  auto sql      = snapshot_visibility_recovery_query(con, vectors) + "0";
  auto native   = run(sql);
  auto expected = ducklake_result_bag(*native);
  auto runtime  = con.context->registered_state->Get<duckdb::SiriusContext>("sirius_state");
  REQUIRE(runtime);
  auto counters         = runtime->physical_counters();
  counters->track_units = true;
  auto hosts =
    runtime->get_memory_manager().get_memory_spaces_for_tier(cucascade::memory::Tier::HOST);
  REQUIRE(hosts.size() == 1);
  auto reserved = hosts[0]->get_total_reserved_memory();
  std::atomic<unsigned> replays{0}, decoded{0};
  std::atomic<uint64_t> reserved_at_replay{UINT64_MAX};
  runtime->cpu_replay_hook_for_testing = [&] {
    ++replays;
    reserved_at_replay = hosts[0]->get_total_reserved_memory();
  };
  counters->parquet_filter_for_testing = [&](bool, uint64_t rows) {
    if (rows) ++decoded;
  };
  std::promise<void> arrived, release;
  auto blocked = arrived.get_future();
  auto resume  = release.get_future().share();
  std::atomic<bool> first{true};
  std::string provider_sql;
  auto observer     = duckdb::make_shared_ptr<provider_query_observer>();
  observer->observe = [&](duckdb::ClientContext& provider) {
    auto state            = duckdb::get_sirius_connection_state(provider);
    auto const& statement = provider.GetCurrentQuery();
    bool selected         = inline_sql ? statement.find(" inlined_data\n") != std::string::npos
                                       : statement.find("data.data_file_id") != std::string::npos;
    if (!selected) return;
    if (!state || state->provenance() != connection_provenance::provider_internal ||
        hosts[0]->get_total_reserved_memory() <= reserved || !first.exchange(false))
      return;
    provider_sql = provider.GetCurrentQuery();
    arrived.set_value();
    if (resume.wait_for(std::chrono::seconds(20)) != std::future_status::ready)
      throw std::runtime_error("snapshot visibility provider rendezvous timed out");
    if (mode != 0) throw duckdb::IOException("injected snapshot visibility provider failure");
  };
  duckdb::OptimizerExtension hook;
  hook.optimizer_info        = observer;
  hook.pre_optimize_function = observe_provider_query;
  duckdb::OptimizerExtension::Register(duckdb::DBConfig::GetConfig(*con.context), std::move(hook));
  auto clear = std::shared_ptr<void>(nullptr, [&](void*) {
    observer->observe                    = {};
    runtime->cpu_replay_hook_for_testing = {};
    counters->parquet_filter_for_testing = {};
  });
  run("SET gpu_execution=true");
  run("SET sirius_test_sync_cpu_replay=true");
  run(std::string("SET enable_duckdb_fallback=") + (mode == 1 ? "false" : "true"));
  auto statement =
    explicit_entry ? "CALL gpu_execution(" + duckdb::Value(sql).ToSQLString() + ")" : sql;
  auto before   = runtime->get_transparent_execution_stats();
  auto active   = std::async(std::launch::async, [&] { return con.Query(statement); });
  bool released = false;
  auto unblock  = std::shared_ptr<void>(nullptr, [&](void*) {
    if (!released) release.set_value();
  });
  REQUIRE(blocked.wait_for(std::chrono::seconds(15)) == std::future_status::ready);
  if (mode == 0) con.Interrupt();
  CHECK(active.wait_for(std::chrono::milliseconds(100)) == std::future_status::timeout);
  release.set_value();
  released = true;
  REQUIRE(active.wait_for(std::chrono::seconds(15)) == std::future_status::ready);
  auto result = active.get();
  INFO(provider_sql);
  INFO((result->HasError() ? result->GetError() : "provider replay succeeded"));
  bool recovered = mode == 2 && explicit_entry;
  REQUIRE(result->HasError() != recovered);
  if (mode == 0) CHECK(result->GetErrorType() == duckdb::ExceptionType::INTERRUPT);
  if (mode == 1) {
    INFO(result->GetError());
    CHECK(result->GetError().find("injected snapshot visibility provider failure") !=
          std::string::npos);
    CHECK(result->GetErrorType() != duckdb::ExceptionType::INTERRUPT);
  }
  if (mode == 2 && !explicit_entry) {
    CHECK(result->GetErrorType() == duckdb::ExceptionType::TRANSACTION);
    CHECK(result->GetError().find("transaction is aborted") != std::string::npos);
  }
  if (mode == 2) CHECK(reserved_at_replay == reserved);
  if (recovered) {
    check_snapshot_result(*result, *native, expected);
    CHECK(reserved_at_replay == reserved);
  }
  CHECK(replays == (mode == 2 ? 1 : 0));
  auto after = runtime->get_transparent_execution_stats();
  CHECK(after.fallbacks == before.fallbacks);
  CHECK(after.runtime_fallbacks == before.runtime_fallbacks + (mode == 2 ? 1 : 0));
  result.reset();
  CHECK(hosts[0]->get_total_reserved_memory() == reserved);
  observer->observe = {};
  run("SET enable_duckdb_fallback=false");
  auto imports = counters->arrow_rows_imported.load();
  auto reads   = decoded.load();
  auto fresh   = run(statement);
  check_snapshot_result(*fresh, *native, expected);
  CHECK(decoded >= reads + 2);
  CHECK(counters->arrow_rows_imported.load() > imports);
  CHECK(replays == (mode == 2 ? 1 : 0));
  CHECK(runtime->get_transparent_execution_stats().runtime_fallbacks == after.runtime_fallbacks);
  fresh.reset();
  CHECK(hosts[0]->get_total_reserved_memory() == reserved);
}

TEST_CASE_METHOD(
  snapshot_fixture,
  "DuckLake snapshot visibility refuses stale data after replacement or lost retention",
  "[isolated_context][ducklake_snapshot_visibility_gpu][ducklake_snapshot_visibility_identity]")
{
  auto vectors        = GENERATE(false, true);
  auto explicit_entry = GENERATE(false, true);
  auto missing        = GENERATE(false, true);
  CAPTURE(vectors, explicit_entry, missing);
  run("SET gpu_execution=false");
  attach_lake(con, disk.directory);
  auto sql      = snapshot_visibility_recovery_query(con, vectors) + "0";
  auto native   = run(sql);
  auto original = ducklake_result_bag(*native);
  auto files    = run("SELECT data_file FROM ducklake_list_files('lake','q')");
  std::string path, field_ids;
  for (duckdb::idx_t i = 0; i < files->RowCount(); ++i) {
    auto candidate = files->Collection().GetValue(0, i).ToString();
    auto schema    = run("SELECT name,field_id FROM parquet_schema(" +
                      duckdb::Value(candidate).ToSQLString() + ") WHERE field_id IS NOT NULL");
    bool partial   = false;
    std::string ids;
    for (duckdb::idx_t column = 0; column < schema->RowCount(); ++column) {
      auto name = schema->Collection().GetValue(0, column).ToString();
      partial |= name == "_ducklake_internal_snapshot_id";
      if (!ids.empty()) ids += ",";
      ids += duckdb::Value(name).ToSQLString() + ":" +
             schema->Collection().GetValue(1, column).ToString();
    }
    if (partial) {
      path      = candidate;
      field_ids = "{" + ids + "}";
      break;
    }
  }
  REQUIRE_FALSE(path.empty());
  auto saved       = (std::filesystem::path(disk.directory) / "original.parquet").string();
  auto replacement = (std::filesystem::path(disk.directory) / "replacement.parquet").string();
  if (!missing)
    run("COPY (SELECT * REPLACE(value+100 AS value) FROM read_parquet(" +
        duckdb::Value(path).ToSQLString() + ")) TO " + duckdb::Value(replacement).ToSQLString() +
        " (FORMAT PARQUET, FIELD_IDS " + field_ids + ")");
  auto runtime = con.context->registered_state->Get<duckdb::SiriusContext>("sirius_state");
  REQUIRE(runtime);
  auto counters         = runtime->physical_counters();
  counters->track_units = true;
  auto hosts =
    runtime->get_memory_manager().get_memory_spaces_for_tier(cucascade::memory::Tier::HOST);
  REQUIRE(hosts.size() == 1);
  auto reserved = hosts[0]->get_total_reserved_memory();
  std::atomic<unsigned> decoded{0}, replays{0};
  counters->parquet_filter_for_testing = [&](bool, uint64_t rows) {
    if (rows) ++decoded;
  };
  runtime->cpu_replay_hook_for_testing = [&] { ++replays; };
  auto clear                           = std::shared_ptr<void>(nullptr, [&](void*) {
    counters->parquet_filter_for_testing = {};
    runtime->cpu_replay_hook_for_testing = {};
  });
  run("SET gpu_execution=true");
  run("SET enable_duckdb_fallback=false");
  run("SET sirius_test_sync_cpu_replay=true");
  auto prepared = con.Prepare(sql);
  REQUIRE_FALSE(prepared->HasError());
  auto execute = [&] {
    if (explicit_entry)
      return con.Query("CALL gpu_execution(" + duckdb::Value(sql).ToSQLString() + ")");
    return prepared->Execute();
  };
  for (int repeat = 0; repeat < 2; ++repeat)
    CHECK(ducklake_result_bag(*execute()) == original);
  REQUIRE(decoded >= 4);
  auto old_source = runtime->get_scan_manager().create_datasource(path);
  REQUIRE(old_source);
  auto old_metadata = old_source->metadata();
  REQUIRE(old_metadata);
  old_source.reset();
  bool moved   = false;
  auto restore = std::shared_ptr<void>(nullptr, [&](void*) {
    if (moved) {
      std::error_code ignored;
      std::filesystem::rename(saved, path, ignored);
    }
  });
  std::vector<std::string> latest;
  if (missing) {
    run("SET gpu_execution=false");
    run("DELETE FROM lake.q WHERE id>=4096 AND id<12288");
    latest = ducklake_result_bag(*run("SELECT value,id FROM lake.q"));
    REQUIRE(latest != original);
  }
  std::filesystem::rename(path, saved);
  moved = true;
  if (!missing) std::filesystem::rename(replacement, path);
  run("SET gpu_execution=false");
  if (missing) {
    // The latest view is still readable: the historical failure must not silently
    // switch to that view when retention of its old data file is lost.
    CHECK(ducklake_result_bag(*run("SELECT value,id FROM lake.q")) == latest);
  }
  auto changed_native = con.Query(sql);
  INFO((changed_native->HasError() ? changed_native->GetError() : "changed native read succeeded"));
  REQUIRE(changed_native->HasError());
  if (!missing) {
    CHECK(changed_native->GetError().find("footer length") != std::string::npos);
    auto source = runtime->get_scan_manager().create_datasource(path);
    REQUIRE(source);
    CHECK(source->metadata() != old_metadata);
  }
  run("SET gpu_execution=true");
  for (bool fallback : {false, true}) {
    INFO(fallback);
    run(std::string("SET enable_duckdb_fallback=") + (fallback ? "true" : "false"));
    auto replay_before = replays.load();
    auto before        = runtime->get_transparent_execution_stats();
    auto result        = execute();
    // The native catalog pins the original footer size. The replacement and
    // missing historical object must both retain errors instead of cached rows.
    REQUIRE(result->HasError());
    if (fallback) CHECK(result->GetErrorType() == changed_native->GetErrorType());
    auto after = runtime->get_transparent_execution_stats();
    CHECK(after.fallbacks == before.fallbacks);
    CHECK(after.runtime_fallbacks - before.runtime_fallbacks == replays.load() - replay_before);
    CHECK(replays.load() - replay_before <= (fallback ? 1 : 0));
    CHECK(replays.load() == replay_before + (fallback ? 1 : 0));
    result.reset();
    CHECK(hosts[0]->get_total_reserved_memory() == reserved);
  }
  std::filesystem::rename(saved, path);
  moved = false;
  run("SET enable_duckdb_fallback=false");
  auto reads         = decoded.load();
  auto replay_before = replays.load();
  auto restored      = execute();
  REQUIRE_FALSE(restored->HasError());
  CHECK(ducklake_result_bag(*restored) == original);
  CHECK(decoded >= reads + 2);
  CHECK(replays == replay_before);
  restored.reset();
  prepared.reset();
  CHECK(hosts[0]->get_total_reserved_memory() == reserved);
}

TEST_CASE_METHOD(
  snapshot_fixture,
  "DuckLake snapshot visibility bounded admission refuses both scans before dependent work",
  "[isolated_context][ducklake_snapshot_visibility_gpu][ducklake_snapshot_visibility_admission]")
{
  auto explicit_entry = GENERATE(false, true);
  CAPTURE(explicit_entry);
  auto fallback = GENERATE(false, true);
  INFO(fallback);
  attach_lake(con, disk.directory);
  run("SET gpu_execution=false");
  auto historical = snapshot_visibility_recovery_query(con, false) + "0";
  std::string sql = "SELECT a.id AS left_id,b.id AS right_id FROM (" + historical + ") a JOIN (" +
                    historical + ") b ON a.id=b.id";
  auto expected = ducklake_result_bag(*run(sql));
  run("SET gpu_execution=true");
  auto runtime = con.context->registered_state->Get<duckdb::SiriusContext>("sirius_state");
  REQUIRE(runtime);
  auto counters         = runtime->physical_counters();
  counters->track_units = true;
  std::atomic<unsigned> decoded{0};
  counters->parquet_filter_for_testing = [&](bool, uint64_t rows) {
    if (rows) ++decoded;
  };
  auto clear_reads =
    std::shared_ptr<void>(nullptr, [&](void*) { counters->parquet_filter_for_testing = {}; });
  auto provider         = std::make_shared<sirius::scan_manager::test::test_reservation_provider>();
  auto envelope         = sirius::ducklake::scan_ingestible::envelope(1, 0);
  provider->grant_bytes = 2 * envelope.retained;
  for (auto bytes : envelope.temporary)
    provider->grant_bytes += bytes;
  counters->preparation_provider_for_testing = provider;
  auto clear                                 = std::shared_ptr<void>(
    nullptr, [&](void*) { counters->preparation_provider_for_testing.reset(); });
  run(std::string("SET enable_duckdb_fallback=") + (fallback ? "true" : "false"));
  auto result = con.Query(
    explicit_entry ? "CALL gpu_execution(" + duckdb::Value(sql).ToSQLString() + ")" : sql);
  if (fallback) {
    REQUIRE_FALSE(result->HasError());
    CHECK(ducklake_result_bag(*result) == expected);
  } else
    CHECK(result->HasError());
  CHECK(provider->seen->requests == 1);
  CHECK(provider->seen->allocations == 0);
  CHECK(provider->outstanding() == 0);
  CHECK(counters->arrow_batches_imported == 0);
  CHECK(decoded == 0);
  result.reset();
  counters->preparation_provider_for_testing.reset();
  run("SET enable_duckdb_fallback=false");
  auto before = runtime->get_transparent_execution_stats();
  auto fresh =
    run(explicit_entry ? "CALL gpu_execution(" + duckdb::Value(sql).ToSQLString() + ")" : sql);
  CHECK(ducklake_result_bag(*fresh) == expected);
  CHECK(decoded >= 2);
  CHECK(runtime->get_transparent_execution_stats().fallbacks == before.fallbacks);
  CHECK(runtime->get_transparent_execution_stats().runtime_fallbacks == before.runtime_fallbacks);
}

TEST_CASE(
  "DuckLake snapshot visibility isolates concurrent prepared visibility and session settings",
  "[isolated_context][ducklake_snapshot_visibility_gpu][ducklake_snapshot_visibility_parallel]")
{
  auto vectors = GENERATE(false, true);
  CAPTURE(vectors);
  if (!sirius::ducklake::bridge_available()) SKIP("DuckLake private binding bridge is disabled");
  sirius::test::scoped_temp_directory disk;
  auto concurrent_config                                = YAML::LoadFile(lake_config().string());
  concurrent_config["sirius"]["max_concurrent_queries"] = 2;
  if (concurrent_config["sirius"]["executor"]["scan_manager"]["max_concurrent_queries"])
    concurrent_config["sirius"]["executor"]["scan_manager"]["max_concurrent_queries"] = 2;
  auto config_path = std::filesystem::path(disk.directory) / "concurrent.yaml";
  {
    std::ofstream output(config_path);
    output << concurrent_config;
  }
  sirius::test::shared_test_env env(config_path);
  auto con = env.make_connection();
  if (!load_ducklake(con)) return;
  auto run = [&](std::string const& sql) { return query(con, sql); };
  attach_lake(con, disk.directory);
  run("SET gpu_execution=false");
  auto prefix = snapshot_visibility_recovery_query(con, vectors);
  std::array<int64_t, 4> limits{0, 1000, 4096, 6000};
  std::vector<std::vector<std::string>> expected;
  for (auto v : limits)
    expected.push_back(
      ducklake_result_bag(*run(prefix + std::to_string(v) + " AND id<" + std::to_string(v + 300))));
  run("SET gpu_execution=true");
  auto runtime = con.context->registered_state->Get<duckdb::SiriusContext>("sirius_state");
  REQUIRE(runtime);
  auto counters         = runtime->physical_counters();
  counters->track_units = true;
  auto hosts =
    runtime->get_memory_manager().get_memory_spaces_for_tier(cucascade::memory::Tier::HOST);
  REQUIRE(hosts.size() == 1);
  auto reserved = hosts[0]->get_total_reserved_memory();
  std::atomic<unsigned> decoded{0};
  counters->parquet_filter_for_testing = [&](bool, uint64_t rows) {
    if (rows) ++decoded;
  };
  auto clear   = std::shared_ptr<void>(nullptr, [&](void*) {
    counters->after_certify_for_testing  = {};
    counters->parquet_filter_for_testing = {};
  });
  auto sibling = env.make_connection();
  run("SET enable_duckdb_fallback=false");
  auto setting = sibling.Query("SET enable_duckdb_fallback=false");
  REQUIRE_FALSE(setting->HasError());
  REQUIRE(runtime->get_config().max_concurrent_queries() == 2);
  std::mutex rendezvous_mutex;
  std::condition_variable rendezvous;
  unsigned arrived = 0;
  std::set<std::thread::id> owners;
  counters->after_certify_for_testing = [&] {
    std::unique_lock lock(rendezvous_mutex);
    if (!owners.insert(std::this_thread::get_id()).second) return;
    ++arrived;
    rendezvous.notify_all();
    if (!rendezvous.wait_for(lock, std::chrono::seconds(15), [&] { return arrived >= 2; }))
      throw std::runtime_error("snapshot visibility queries did not reach capture concurrently");
  };
  run("SET scan_task_batch_size=1");
  query(sibling, "SET scan_task_batch_size=4096");
  auto before = runtime->get_transparent_execution_stats();
  std::promise<void> release;
  auto ready   = release.get_future().share();
  auto execute = [&](duckdb::Connection& con, int shift) {
    ready.wait();
    auto prepared = con.Prepare(prefix + "? AND id<?");
    if (prepared->HasError()) throw std::runtime_error(prepared->GetError());
    for (int repeat = 0; repeat < 8; ++repeat) {
      auto index  = (repeat + shift) % limits.size();
      auto v      = limits[index];
      auto result = prepared->Execute(v, v + 300);
      if (ducklake_result_bag(*result) != expected[index]) return false;
    }
    return true;
  };
  auto a = std::async(std::launch::async, [&] { return execute(con, 0); });
  auto b = std::async(std::launch::async, [&] { return execute(sibling, 2); });
  release.set_value();
  REQUIRE(a.wait_for(std::chrono::seconds(30)) == std::future_status::ready);
  REQUIRE(b.wait_for(std::chrono::seconds(30)) == std::future_status::ready);
  CHECK(a.get());
  CHECK(b.get());
  auto after = runtime->get_transparent_execution_stats();
  CHECK(after.executions == before.executions + 16);
  CHECK(after.fallbacks == before.fallbacks);
  CHECK(after.runtime_fallbacks == before.runtime_fallbacks);
  CHECK(arrived == 2);
  CHECK(decoded >= 16);
  CHECK(hosts[0]->get_total_reserved_memory() == reserved);
  CHECK(run("SELECT current_setting('scan_task_batch_size')")
          ->Collection()
          .GetValue(0, 0)
          .GetValue<uint64_t>() == 1);
  CHECK(query(sibling, "SELECT current_setting('scan_task_batch_size')")
          ->Collection()
          .GetValue(0, 0)
          .GetValue<uint64_t>() == 4096);
}

TEST_CASE("DuckLake snapshot visibility joint visibility executes and recovers across two GPUs",
          "[isolated_context][ducklake_snapshot_visibility_multi_gpu][multi_gpu]")
{
  if (!sirius::ducklake::bridge_available()) SKIP("DuckLake private binding bridge is disabled");
  if (!sirius::test::has_gpus(2)) SKIP("DuckLake device placement requires two GPUs");
  sirius::test::scoped_temp_directory disk;
  auto config                              = YAML::LoadFile(lake_config().string());
  config["sirius"]["topology"]["num_gpus"] = 2;
  auto config_path                         = std::filesystem::path(disk.directory) / "devices.yaml";
  {
    std::ofstream output(config_path);
    output << config;
  }
  sirius::test::shared_test_env env(config_path);
  auto con = env.make_connection();
  if (!load_ducklake(con)) return;
  auto run = [&](std::string const& sql) { return query(con, sql); };
  attach_lake(con, disk.directory);
  run("SET gpu_execution=false");
  auto vectors = GENERATE(false, true);
  CAPTURE(vectors);
  auto historical = snapshot_visibility_recovery_query(con, vectors, 327680) + "0";
  std::string sql = "SELECT value,COUNT(*),SUM(id) FROM (" + historical + ") q GROUP BY value";
  run("SET scan_task_batch_size=4096");
  auto expected = ducklake_result_bag(*run(sql));
  run("SET gpu_execution=true");
  auto runtime = con.context->registered_state->Get<duckdb::SiriusContext>("sirius_state");
  REQUIRE(runtime);
  auto counters         = runtime->physical_counters();
  counters->track_units = true;
  std::mutex mutex;
  std::set<int> devices;
  bool inject = false, fired = false;
  counters->parquet_filter_for_testing = [&](bool, uint64_t rows) {
    if (!rows) return;
    int device = -1;
    if (cudaGetDevice(&device) != cudaSuccess)
      throw std::runtime_error("cannot observe GPU device");
    std::lock_guard lock(mutex);
    devices.insert(device);
    if (inject && devices.size() == 2 && !std::exchange(fired, true))
      throw sirius::transparent::classified_execution_error(
        sirius::transparent::late_failure_cause::reader_io,
        "DuckLake snapshot visibility failure after both devices decoded");
  };
  auto clear =
    std::shared_ptr<void>(nullptr, [&](void*) { counters->parquet_filter_for_testing = {}; });
  auto hosts =
    runtime->get_memory_manager().get_memory_spaces_for_tier(cucascade::memory::Tier::HOST);
  REQUIRE(hosts.size() == 1);
  auto reserved = hosts[0]->get_total_reserved_memory();
  for (bool explicit_entry : {false, true}) {
    INFO(explicit_entry);
    for (bool fail : {false, true, false}) {
      INFO(fail);
      devices.clear();
      inject = fail;
      fired  = false;
      run(std::string("SET enable_duckdb_fallback=") + (fail ? "true" : "false"));
      auto before   = runtime->get_transparent_execution_stats();
      auto imported = counters->arrow_rows_imported.load();
      auto result =
        run(explicit_entry ? "CALL gpu_execution(" + duckdb::Value(sql).ToSQLString() + ")" : sql);
      CHECK(ducklake_result_bag(*result) == expected);
      result.reset();
      CHECK(devices.size() == 2);
      CHECK(fired == fail);
      auto after = runtime->get_transparent_execution_stats();
      if (!explicit_entry) {
        CHECK(after.runtime_fallbacks == before.runtime_fallbacks + (fail ? 1 : 0));
        CHECK(after.executions == before.executions + 1);
      }
      if (!fail) CHECK(counters->arrow_rows_imported.load() > imported);
      CHECK(hosts[0]->get_total_reserved_memory() == reserved);
    }
  }
}

TEST_CASE_METHOD(ducklake_fixture,
                 "DuckLake bound source policy sees S3 deletes beside local data",
                 "[isolated_context][ducklake][ducklake_source_policy]")
{
  if (!sirius::ducklake::bridge_available()) SKIP("DuckLake private binding bridge is disabled");
  run("SET gpu_execution=false");
  attach_lake(con, disk.directory);
  run("CALL lake.set_option('data_inlining_row_limit', 0)");
  run("CREATE TABLE lake.q AS SELECT i::BIGINT id FROM range(100) r(i)");
  run("DELETE FROM lake.q WHERE id % 2 = 0");
  auto deletes = run(
    "SELECT count(*) FROM __ducklake_metadata_lake.ducklake_delete_file "
    "WHERE end_snapshot IS NULL");
  REQUIRE(deletes->Collection().GetValue(0, 0).GetValue<int64_t>() > 0);
  run(
    "UPDATE __ducklake_metadata_lake.ducklake_delete_file "
    "SET path='s3://phase3-policy-test/deletes.parquet', path_is_relative=false "
    "WHERE end_snapshot IS NULL");
  bool observed          = false;
  auto observer          = duckdb::make_shared_ptr<provider_query_observer>();
  observer->inspect_plan = [&](duckdb::ClientContext& context, duckdb::LogicalOperator& plan) {
    if (&context != con.context.get() || observed) return;
    auto policy = sirius::transparent::derive_plan_source_policy(plan, context);
    REQUIRE(policy.discovery_complete);
    REQUIRE(policy.scans.size() == 1);
    CHECK(policy.reads_sirius_owned_s3());
    CHECK_FALSE(policy.cpu_replay_permitted());
    CHECK_THROWS_WITH(
      sirius::transparent::require_cpu_replay(policy, "SELECT * FROM lake.q", "test GPU refusal"),
      Catch::Matchers::ContainsSubstring("S3 CPU fallback is not supported"));
    observed = true;
    // Stop before execution: this test must not open the deliberately absent S3 object.
    throw std::runtime_error("phase3 source policy inspected");
  };
  duckdb::OptimizerExtension hook;
  hook.optimizer_info        = observer;
  hook.pre_optimize_function = observe_provider_query;
  duckdb::OptimizerExtension::Register(duckdb::DBConfig::GetConfig(*con.context), std::move(hook));
  auto clear  = std::shared_ptr<void>(nullptr, [&](void*) { observer->inspect_plan = {}; });
  auto result = con.Query("SELECT * FROM lake.q");
  REQUIRE(result);
  CHECK(result->HasError());
  CHECK(result->GetError().find("phase3 source policy inspected") != std::string::npos);
  CHECK(observed);
  observer->inspect_plan = {};
  run("SET gpu_execution=true");
  auto refused = con.Query("SELECT * FROM lake.q");
  REQUIRE(refused);
  REQUIRE(refused->HasError());
  CHECK(refused->GetError().find("S3 CPU fallback is not supported") != std::string::npos);
}

namespace {
std::filesystem::path remote_delete_cache_config(std::string const& directory)
{
  auto config = YAML::LoadFile(lake_config().string());
  config["sirius"]["executor"]["scan_manager"]["cache"]["mode"]     = "sirius";
  config["sirius"]["executor"]["scan_manager"]["cache"]["eviction"] = "lru";
  // Exercise auth and service retry budgets separately without long backoffs.
  auto rest                       = config["sirius"]["executor"]["scan_manager"]["rest"];
  rest["max_auth_retry_attempts"] = 2;
  rest["max_retry_attempts"]      = 3;
  rest["retry_backoff_base_ms"]   = 0;
  rest["retry_jitter_ms"]         = 0;
  auto path                       = std::filesystem::path(directory) / "remote-cache.yaml";
  std::ofstream output(path);
  output << config;
  return path;
}
std::unique_ptr<cucascade::io::datasource> cache_remote_delete(duckdb::Connection& con,
                                                               duckdb::SiriusContext& runtime,
                                                               std::string const& uri,
                                                               std::vector<uint8_t> const& expected)
{
  auto context = sirius::io::s3::capture_duckdb_s3_context(
    *con.context,
    runtime.get_scan_manager(),
    uri,
    runtime.get_config().get_scan_manager_config().object_store);
  auto source = sirius::io::open_datasource(context, uri);
  REQUIRE(source->uses_fs_cache());
  std::vector<cudf::io::text::byte_range_info> ranges;
  ranges.emplace_back(0, static_cast<int64_t>(expected.size()));
  source->fadvise(ranges, 0);
  REQUIRE(source->prepare_prefetch(false) == cucascade::io::prepare_result::prepared);
  std::vector<uint8_t> actual(expected.size());
  REQUIRE(source->host_read(0, actual.size(), actual.data()) == actual.size());
  REQUIRE(actual == expected);
  return source;  // Keep the old generation resident while the query opens again.
}
}  // namespace

TEST_CASE("DuckLake REST deletes preserve bound snapshot visibility",
          "[.][isolated_context][ducklake][ducklake_remote_deletes][s3]")
{
  if (!sirius::ducklake::bridge_available()) SKIP("DuckLake private binding bridge is disabled");
  auto httpfs = sirius::test::s3::env_or("SIRIUS_TEST_DUCKLAKE_HTTPFS_EXTENSION");
  REQUIRE_FALSE(httpfs.empty());
  {
    sirius::util::env_guard disabled("SIRIUS_DISABLE", std::string("1"));
    REQUIRE(sirius::test::ensure_s3_test_env());
  }
  auto remote_data      = GENERATE(false, true);
  auto puffin           = GENERATE(true, false);
  auto snapshot_deletes = GENERATE(false, true);
  CAPTURE(remote_data, puffin, snapshot_deletes);
  sirius::test::scoped_temp_directory disk;
  auto bucket = sirius::test::s3::require_env("SIRIUS_TEST_S3_BUCKET");
  auto prefix =
    "s3://" + bucket + (remote_data ? "/ducklake-native-remote/" : "/ducklake-native-local/");
  auto quote = [](std::string const& value) { return duckdb::Value(value).ToSQLString(); };
  auto credentials =
    ", KEY_ID " + quote(sirius::test::s3::require_env("SIRIUS_TEST_S3_ACCESS_KEY")) + ", SECRET " +
    quote(sirius::test::s3::require_env("SIRIUS_TEST_S3_SECRET_KEY")) + ", REGION " +
    quote(sirius::test::s3::env_or("SIRIUS_TEST_S3_REGION", "us-east-1"));
  std::vector<std::string> statements;
  std::vector<std::string> script;
  std::string delete_key, bare_data_path;
  std::vector<uint8_t> delete_bytes;
  using result_bag = decltype(ducklake_result_bag(std::declval<duckdb::QueryResult&>()));
  std::vector<result_bag> expected;
  std::vector<duckdb::vector<duckdb::LogicalType>> expected_types;
  std::vector<duckdb::vector<duckdb::Identifier>> expected_names;
  auto prepare = "PREPARE remote_q AS SELECT id FROM lake.q WHERE id >= $1";
  auto attach  = "ATTACH 'ducklake:" + disk.directory + "/meta.ducklake' AS lake (DATA_PATH " +
                quote(remote_data ? prefix : disk.directory + "/data/") + ")";
  {
    // Native writer and oracle have their own database and httpfs. Do not load
    // Sirius or bypass its GPU-only S3 filesystem policy.
    duckdb::DBConfig config;
    config.options.load_extensions = false;
    config.SetOptionByName("allow_unsigned_extensions", duckdb::Value::BOOLEAN(true));
    duckdb::DuckDB native_db(nullptr, &config);
    duckdb::ExtensionHelper::LoadExtension(native_db, "parquet");
    duckdb::ExtensionHelper::LoadExtension(native_db, "core_functions");
    duckdb::Connection native(native_db);
    query(native, "LOAD " + quote(httpfs));
    REQUIRE(load_ducklake(native));
    auto endpoint = sirius::test::s3::require_env("SIRIUS_TEST_S3_ENDPOINT");
    REQUIRE(endpoint.starts_with("http://"));
    endpoint.erase(0, 7);
    query(native,
          "CREATE SECRET native_remote (TYPE S3, SCOPE " + quote(prefix) + credentials +
            ", ENDPOINT " + quote(endpoint) + ", URL_STYLE 'path', USE_SSL false)");
    query(native, attach);
    query(native, "CALL lake.set_option('data_inlining_row_limit',0)");
    query(native,
          "CALL lake.set_option('write_deletion_vectors'," +
            std::string(puffin ? "true" : "false") + ")");
    query(native, "CREATE TABLE lake.q AS SELECT i::BIGINT id FROM range(128) r(i)");
    auto current_snapshot = [&] {
      return query(native, "SELECT max(snapshot_id) FROM ducklake_snapshots('lake')")
        ->Collection()
        .GetValue(0, 0)
        .GetValue<uint64_t>();
    };
    auto initial_snapshot = current_snapshot();
    if (snapshot_deletes) {
      query(native, "INSERT INTO lake.q SELECT i FROM range(128,256) r(i)");
      REQUIRE(query(native, "CALL ducklake_merge_adjacent_files('lake','q')")->RowCount() == 1);
      REQUIRE(query(native,
                    "SELECT count(*) FROM __ducklake_metadata_lake.ducklake_data_file "
                    "WHERE end_snapshot IS NULL AND partial_max IS NOT NULL")
                ->Collection()
                .GetValue(0, 0)
                .GetValue<int64_t>() == 1);
    }
    query(native, "DELETE FROM lake.q WHERE id=1");
    auto snapshot = current_snapshot();
    query(native, "DELETE FROM lake.q WHERE id=2");
    auto deleted_snapshot = current_snapshot();
    auto delete_profile =
      query(native,
            "SELECT format,partial_max FROM __ducklake_metadata_lake.ducklake_delete_file "
            "WHERE end_snapshot IS NULL");
    REQUIRE(delete_profile->RowCount() == 1);
    REQUIRE(delete_profile->Collection().GetValue(0, 0).ToString() ==
            (puffin ? "puffin" : "parquet"));
    if (snapshot_deletes) REQUIRE_FALSE(delete_profile->Collection().GetValue(1, 0).IsNull());
    // The same scan now owns both remote file visibility and committed inline rows.
    query(native, "CALL lake.set_option('data_inlining_row_limit',100)");
    query(native, "INSERT INTO lake.q VALUES (20000),(20000),(20001)");
    if (!remote_data) {
      // Preserve every historical delete object, including older Parquet files.
      auto files = query(native,
                         "SELECT delete_file_id,begin_snapshot FROM "
                         "__ducklake_metadata_lake.ducklake_delete_file");
      REQUIRE(files->RowCount() >= 1);
      for (duckdb::idx_t row = 0; row < files->RowCount(); ++row) {
        auto id       = files->Collection().GetValue(0, row).ToString();
        auto version  = files->Collection().GetValue(1, row).ToString();
        auto resolved = query(native,
                              "SELECT delete_file FROM ducklake_list_files('lake','q', "
                              "snapshot_version=>" +
                                version + ")");
        REQUIRE(resolved->RowCount() == 1);
        auto path = resolved->Collection().GetValue(0, 0).ToString();
        CAPTURE(path);
        std::ifstream input(path, std::ios::binary);
        REQUIRE(input.good());
        std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(input)), {});
        auto name = "delete-" + id + (puffin ? ".puffin" : ".parquet");
        REQUIRE(sirius::test::put_s3_test_object("ducklake-native-local/" + name, bytes));
        query(native,
              "UPDATE __ducklake_metadata_lake.ducklake_delete_file SET path=" +
                quote(prefix + name) + ", path_is_relative=false WHERE delete_file_id=" + id);
      }
    }
    std::string bare_before;
    if (puffin) {
      query(native, "CALL lake.set_option('data_inlining_row_limit',0)");
      query(native, "CREATE TABLE lake.bare_q AS SELECT i::BIGINT id FROM range(16) r(i)");
      bare_before = std::to_string(current_snapshot());
      if (snapshot_deletes) {
        query(native, "INSERT INTO lake.bare_q SELECT i FROM range(16,32) r(i)");
        REQUIRE(query(native, "CALL ducklake_merge_adjacent_files('lake','bare_q')")->RowCount() ==
                1);
      }
      query(native, "DELETE FROM lake.bare_q WHERE id=1");
      auto bare_file =
        query(native, "SELECT delete_file,data_file FROM ducklake_list_files('lake','bare_q')");
      REQUIRE(bare_file->RowCount() == 1);
      auto bare_path = bare_file->Collection().GetValue(0, 0).ToString();
      bare_data_path = bare_file->Collection().GetValue(1, 0).ToString();
      native.BeginTransaction();
      auto handle = duckdb::FileSystem::GetFileSystem(*native.context)
                      .OpenFile(bare_path, duckdb::FileFlags::FILE_FLAGS_READ);
      std::vector<uint8_t> bytes(handle->GetFileSize());
      REQUIRE(bytes.size() >= 12);
      handle->Read(bytes.data(), bytes.size(), 0);
      handle.reset();
      native.Commit();
      std::array<uint8_t, 4> magic{0xd1, 0xd3, 0x39, 0x64};
      REQUIRE(std::equal(magic.begin(), magic.end(), bytes.begin() + 4));
      uint64_t length = (uint64_t(bytes[0]) << 24) | (uint64_t(bytes[1]) << 16) |
                        (uint64_t(bytes[2]) << 8) | bytes[3];
      REQUIRE(length + 8 == bytes.size());
      if (!remote_data) {
        REQUIRE(sirius::test::put_s3_test_object("ducklake-native-local/bare.puffin", bytes));
        query(native,
              "UPDATE __ducklake_metadata_lake.ducklake_delete_file SET path=" +
                quote(prefix + "bare.puffin") + ", path_is_relative=false WHERE ends_with(" +
                quote(bare_path) + ", path)");
      }
      auto remote_bare =
        query(native, "SELECT delete_file FROM ducklake_list_files('lake','bare_q')");
      REQUIRE(remote_bare->Collection().GetValue(0, 0).ToString().starts_with(prefix));
      query(native, "CALL lake.set_option('data_inlining_row_limit',100)");
    }
    auto at = [](uint64_t version) {
      return "lake.q AT (VERSION => " + std::to_string(version) + ")";
    };
    statements = {"SELECT id FROM lake.q",
                  "SELECT id FROM " + at(snapshot),
                  "SELECT id FROM " + at(initial_snapshot),
                  "SELECT id FROM " + at(deleted_snapshot),
                  "SELECT a.id AS old_id,b.id AS current_id FROM (SELECT id FROM " +
                    at(initial_snapshot) + ") a JOIN lake.q b ON a.id=b.id WHERE a.id<8",
                  "SELECT id FROM lake.q WHERE id>=120 ORDER BY id LIMIT 3",
                  "EXECUTE remote_q(120)",
                  "EXECUTE remote_q(0)",
                  "EXECUTE remote_q(20000)"};
    if (puffin) {
      statements.push_back("SELECT id FROM lake.bare_q");
      statements.push_back("SELECT id FROM lake.bare_q AT (VERSION => " + bare_before + ")");
    }
    if (snapshot_deletes && !puffin) {
      auto file = query(native, "SELECT delete_file FROM ducklake_list_files('lake','q')");
      REQUIRE(file->RowCount() == 1);
      auto snapshots = query(native,
                             "SELECT count(DISTINCT _ducklake_internal_snapshot_id), "
                             "min(_ducklake_internal_snapshot_id), "
                             "max(_ducklake_internal_snapshot_id) FROM read_parquet(" +
                               quote(file->Collection().GetValue(0, 0).ToString()) + ")");
      REQUIRE(snapshots->Collection().GetValue(0, 0).GetValue<int64_t>() == 2);
      REQUIRE(snapshots->Collection().GetValue(1, 0).GetValue<uint64_t>() == snapshot);
      REQUIRE(snapshots->Collection().GetValue(2, 0).GetValue<uint64_t>() == deleted_snapshot);
    }
    auto delete_file = query(native, "SELECT delete_file FROM ducklake_list_files('lake','q')");
    REQUIRE(delete_file->RowCount() == 1);
    auto delete_uri    = delete_file->Collection().GetValue(0, 0).ToString();
    auto bucket_prefix = "s3://" + bucket + "/";
    REQUIRE(delete_uri.starts_with(bucket_prefix));
    delete_key = delete_uri.substr(bucket_prefix.size());
    native.BeginTransaction();
    auto handle = duckdb::FileSystem::GetFileSystem(*native.context)
                    .OpenFile(delete_uri, duckdb::FileFlags::FILE_FLAGS_READ);
    delete_bytes.resize(handle->GetFileSize());
    REQUIRE_FALSE(delete_bytes.empty());
    handle->Read(delete_bytes.data(), delete_bytes.size(), 0);
    handle.reset();
    native.Commit();
    // Close the native metadata database before copying. Both executions start
    // from identical committed metadata; their later commits cannot affect each other.
    query(native, "DETACH lake");
    std::filesystem::copy_file(disk.directory + "/meta.ducklake", disk.directory + "/gpu.ducklake");
    query(native, attach);
    auto append_views = [&] {
      for (int repeat = 0; repeat < 2; ++repeat)
        script.insert(script.end(), statements.begin(), statements.end());
    };
    append_views();
    if (snapshot_deletes) {
      for (bool delete_file : {false, true}) {
        // Sirius's S3 filesystem is read-only. Exercise new transaction delete
        // files on local data, alongside historical remote delete dependencies.
        if (delete_file && remote_data) continue;
        if (delete_file) script.push_back("CALL lake.set_option('data_inlining_row_limit',0)");
        for (bool commit : {false, true}) {
          script.push_back("BEGIN");
          // Delete one file row and committed inline rows in each mode.
          script.push_back(delete_file ? "DELETE FROM lake.q WHERE id IN (4,20001)"
                                       : "DELETE FROM lake.q WHERE id IN (3,20000)");
          append_views();
          script.push_back(commit ? "COMMIT" : "ROLLBACK");
          append_views();
        }
      }
    }
    query(native, prepare);
    for (auto const& sql : script) {
      auto result = query(native, sql);
      expected_types.push_back(result->GetTypes());
      expected_names.push_back(result->GetNames());
      expected.push_back(ducklake_result_bag(*result));
    }
    query(native, "DETACH lake");
  }
  REQUIRE(expected.at(0).size() == (snapshot_deletes ? 257 : 129));
  REQUIRE(expected.at(1).size() == (snapshot_deletes ? 255 : 127));
  REQUIRE(expected.at(2).size() == 128);
  REQUIRE(expected.at(3).size() == (snapshot_deletes ? 254 : 126));
  REQUIRE(expected.at(4).size() == 6);
  REQUIRE(expected.at(8).size() == 3);
  if (puffin) {
    REQUIRE(expected.at(9).size() == (snapshot_deletes ? 31 : 15));
    REQUIRE(expected.at(10).size() == 16);
  }
  sirius::test::shared_test_env env(remote_delete_cache_config(disk.directory));
  auto con = env.make_connection();
  REQUIRE(load_ducklake(con));
  query(con,
        "CREATE SECRET ducklake_remote (TYPE SIRIUS_S3, SCOPE " + quote(prefix) + credentials +
          ", ENDPOINT " + quote(sirius::test::s3::require_env("SIRIUS_TEST_S3_ENDPOINT")) + ")");
  auto gpu_attach = duckdb::StringUtil::Replace(attach, "/meta.ducklake", "/gpu.ducklake");
  query(con, gpu_attach);
  query(con, "SET gpu_execution=true");
  auto runtime = con.context->registered_state->Get<duckdb::SiriusContext>("sirius_state");
  REQUIRE(runtime);
  REQUIRE(runtime->get_scan_manager().planning_io_backend(prefix + "probe.puffin") ==
          cucascade::io::io_context_type::restful);
  query(con, "SET parquet_metadata_cache=true");
  query(con, "SET validate_external_file_cache='NO_VALIDATION'");
  query(con, "SET enable_duckdb_fallback=true");
  // Keep the locator and bound size unchanged. A new open must see the new
  // bytes, even with native metadata caching and NO_VALIDATION requested.
  auto check_replacement = [&](bool warm) {
    CAPTURE(warm);
    auto cached =
      warm ? cache_remote_delete(con, *runtime, "s3://" + bucket + "/" + delete_key, delete_bytes)
           : nullptr;
    std::vector<uint8_t> corrupt(delete_bytes.size(), 0);
    REQUIRE(sirius::test::put_s3_test_object(delete_key, corrupt));
    auto before   = runtime->get_transparent_execution_stats();
    auto failed   = con.Query(statements.at(3));
    bool restored = sirius::test::put_s3_test_object(delete_key, delete_bytes);
    REQUIRE(restored);
    REQUIRE(failed);
    REQUIRE(failed->HasError());
    CHECK(failed->GetError().find("S3 CPU fallback is not supported") != std::string::npos);
    auto after = runtime->get_transparent_execution_stats();
    CHECK(after.fallbacks == before.fallbacks);
    CHECK(after.runtime_fallbacks == before.runtime_fallbacks);
    auto recovered = query(con, statements.at(3));
    CHECK(ducklake_result_bag(*recovered) == expected.at(3));
  };
  check_replacement(false);  // No DV payload has been read by this runtime.
  check_replacement(true);   // Explicitly prefetch and retain the original generation.
  query(con, prepare);
  // Exercise the real child factory and preparation path after another owner
  // changes the manager's URI map. A captured test-only resolver would hide the bug.
  bool scope_checked       = false;
  bool transaction_deletes = false, local_binding_checked = false;
  bool transaction_delete_file = false;
  auto observer                = duckdb::make_shared_ptr<provider_query_observer>();
  observer->inspect_plan = [&](duckdb::ClientContext& context, duckdb::LogicalOperator& plan) {
    if (&context != con.context.get() ||
        (scope_checked && (!transaction_deletes || local_binding_checked)))
      return;
    auto visit = [&](auto&& self, duckdb::LogicalOperator& op) -> void {
      if (op.type == duckdb::LogicalOperatorType::LOGICAL_GET) {
        auto& get = op.Cast<duckdb::LogicalGet>();
        if (get.function.GetName() == "ducklake_scan") {
          namespace dl = sirius::ducklake;
          namespace sm = sirius::scan_manager;
          auto binding = dl::binding_handle::capture(get.function, get.bind_data.get(), context);
          binding->bind_execution_owner();
          binding->configure_scan({0}, nullptr);
          uint64_t index = 0;
          auto unit      = binding->unit_at(index);
          while (unit && unit->kind != dl::unit_kind::parquet)
            unit = binding->unit_at(++index);
          REQUIRE(unit);
          if (scope_checked) {
            // The script also visits bare_q; its fixture has no transaction deletes.
            if (unit->input.path == bare_data_path) return;
            CHECK(unit->transaction_local_delete == transaction_delete_file);
            if (transaction_delete_file) {
              CHECK_FALSE(unit->delete_path.starts_with("s3://"));
            } else {
              CHECK(unit->inline_delete_count == 1);
              CHECK(unit->delete_path.starts_with("s3://"));
            }
            CHECK(unit->deletes ==
                  (puffin ? dl::delete_format::puffin : dl::delete_format::parquet));
            local_binding_checked = true;
            return;
          }
          scope_checked = true;
          auto envelope = dl::parquet_unit_ingestible::envelope(*unit);
          sm::test::test_reservation_provider provider;
          provider.grant_bytes = 128 * 1024 * 1024;
          auto unique          = std::make_unique<sm::preparation_ledger>(provider);
          std::array scans{sm::scan_envelope{{envelope}, {}, true}};
          auto decision =
            unique->admit(scans, sm::memory_space_id(cucascade::memory::Tier::HOST, 0));
          REQUIRE(decision.deferred);
          auto admission = std::make_shared<sm::preparation_admission>(std::move(unique), decision);
          auto ledger    = std::shared_ptr<sm::preparation_ledger>(admission->consume());
          auto child     = binding->make_parquet_unit(index, 1, 1, admission, {0});
          child->attach_preparation_ledger(ledger);
          child->preflight_for_query();
          auto& manager = runtime->get_scan_manager();
          auto good     = sirius::io::s3::resolve_duckdb_s3_secret(
            context,
            unit->delete_path,
            runtime->get_config().get_scan_manager_config().object_store);
          auto restore = std::shared_ptr<void>(nullptr, [&](void*) {
            child->finish_preparation();
            ledger->close();
            if (remote_data) manager.install_s3_config(unit->input.path, good);
            manager.install_s3_config(unit->delete_path, good);
          });
          // Same key and size, but an unusable access scope. The existing task
          // must retain its valid context rather than reopening through this map.
          std::async(std::launch::async, [&] {
            auto changed     = good;
            changed.endpoint = "http://127.0.0.1:1";
            if (remote_data) manager.install_s3_config(unit->input.path, changed);
            manager.install_s3_config(unit->delete_path, changed);
          }).get();
          auto sources = child->make_preparation_sources(
            [&](std::string_view path) { return manager.create_datasource(path)->io_ctx(); });
          size_t prepared = 0;
          for (auto& source : sources) {
            auto claim = source.try_claim();
            REQUIRE(std::holds_alternative<sm::preparation_job>(claim));
            auto metadata = std::get<sm::preparation_job>(std::move(claim)).run({});
            if (!metadata) continue;
            auto const& visibility =
              dynamic_cast<dl::dv_visibility const&>(*metadata->dependencies().front().visibility);
            if (snapshot_deletes && puffin) CHECK(visibility.validated_blobs == 2);
            REQUIRE(visibility.positions.size() == 2);
            CHECK(visibility.positions[0] == 1);
            CHECK(visibility.positions[1] == 2);
            ++prepared;
          }
          CHECK(prepared == 1);
        }
      }
      for (auto& child : op.children)
        self(self, *child);
    };
    visit(visit, plan);
  };
  duckdb::OptimizerExtension scope_hook;
  scope_hook.optimizer_info        = observer;
  scope_hook.pre_optimize_function = observe_provider_query;
  duckdb::OptimizerExtension::Register(duckdb::DBConfig::GetConfig(*con.context),
                                       std::move(scope_hook));
  auto clear_observer = std::shared_ptr<void>(nullptr, [&](void*) { observer->inspect_plan = {}; });
  auto counters       = runtime->physical_counters();
  counters->track_units = true;
  std::atomic<unsigned> decoded{0};
  counters->parquet_filter_for_testing = [&](bool, uint64_t rows) {
    if (rows) ++decoded;
  };
  auto clear_decode =
    std::shared_ptr<void>(nullptr, [&](void*) { counters->parquet_filter_for_testing = {}; });
  query(con, "SET enable_duckdb_fallback=false");
  // The script repeats views before, during and after rollback/commit. DML is
  // performed by DuckLake; each SELECT/EXECUTE must still run through Sirius.
  for (size_t i = 0; i < script.size(); ++i) {
    CAPTURE(i, script[i]);
    auto before = runtime->get_transparent_execution_stats();
    auto reads  = decoded.load();
    auto gpu    = query(con, script[i]);
    if (script[i] == "CALL lake.set_option('data_inlining_row_limit',0)")
      transaction_delete_file = true;
    if (script[i] == "BEGIN") local_binding_checked = false;
    if (script[i].starts_with("DELETE")) transaction_deletes = true;
    if (script[i] == "COMMIT" || script[i] == "ROLLBACK") {
      CHECK(local_binding_checked);
      transaction_deletes = false;
    }
    CHECK(gpu->GetTypes() == expected_types[i]);
    CHECK(gpu->GetNames() == expected_names[i]);
    CHECK(ducklake_result_bag(*gpu) == expected[i]);
    if (script[i].starts_with("SELECT") || script[i].starts_with("EXECUTE")) {
      auto after = runtime->get_transparent_execution_stats();
      CHECK(after.executions == before.executions + 1);
      CHECK(after.fallbacks == before.fallbacks);
      CHECK(after.runtime_fallbacks == before.runtime_fallbacks);
      // The inline-only parameter may legitimately prune every Parquet unit.
      if (script[i] != "EXECUTE remote_q(20000)") CHECK(decoded.load() > reads);
    }
  }
  CHECK(scope_checked);
}

TEST_CASE("DuckLake remote delete queries enforce backend identity contracts",
          "[.][isolated_context][ducklake_remote_identity]")
{
  if (!sirius::ducklake::bridge_available()) SKIP("DuckLake private binding bridge is disabled");
  auto puffin = GENERATE(false, true);
  auto warm   = GENERATE(false, true);
  // Missing/weak ETag, changed response, 412, 403, 503, cancellation, allocation failure.
  auto fault_kind = GENERATE(0, 1, 2, 3, 4, 5, 6, 7);
  CAPTURE(puffin, warm, fault_kind);
  sirius::test::scoped_temp_directory disk;
  sirius::test::shared_test_env env(remote_delete_cache_config(disk.directory));
  auto con = env.make_connection();
  REQUIRE(load_ducklake(con));
  auto run = [&](std::string const& sql) { return query(con, sql); };
  run("SET gpu_execution=false");
  attach_lake(con, disk.directory);
  run("CALL lake.set_option('data_inlining_row_limit',0)");
  run(std::string("CALL lake.set_option('write_deletion_vectors',") +
      (puffin ? "true)" : "false)"));
  run("CREATE TABLE lake.q AS SELECT i::BIGINT id FROM range(128) r(i)");
  run("DELETE FROM lake.q WHERE id IN (1,2)");
  std::string sql = "SELECT id FROM lake.q";
  auto expected   = ducklake_result_bag(*run(sql));
  REQUIRE(expected.size() == 126);
  auto files = run("SELECT delete_file FROM ducklake_list_files('lake','q')");
  REQUIRE(files->RowCount() == 1);
  std::ifstream input(files->Collection().GetValue(0, 0).ToString(), std::ios::binary);
  REQUIRE(input.good());
  std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(input)), {});
  cucascade::test::range_fault_policy good;
  good.successful_head_etag = good.successful_get_etag = "\"opened\"";
  auto fault                                           = good;
  if (fault_kind == 0) fault.successful_head_etag.clear();
  if (fault_kind == 1) fault.successful_head_etag = "W/\"opened\"";
  if (fault_kind == 2) fault.successful_get_etag = "\"replaced\"";
  if (fault_kind >= 3 && fault_kind <= 5) {
    fault.fail_all_gets = true;
    fault.fail_status   = fault_kind == 3 ? 412 : (fault_kind == 4 ? 403 : 503);
  }
  auto response = std::make_shared<cucascade::test::range_response_control>();
  if (fault_kind == 6) {
    fault.body.control          = response;
    fault.body.hold_after_bytes = 0;
  }
  cucascade::test::loopback_range_server healthy(bytes, good), failing(bytes, fault);
  std::string uri = "s3://identity-bucket/delete";
  run("UPDATE __ducklake_metadata_lake.ducklake_delete_file SET path='" + uri +
      "', path_is_relative=false WHERE end_snapshot IS NULL");
  auto endpoint = [&](std::string const& value) {
    run(
      "CREATE OR REPLACE SECRET identity_remote (TYPE SIRIUS_S3, SCOPE 's3://identity-bucket/', "
      "KEY_ID 'access', SECRET 'secret', REGION 'us-east-1', ENDPOINT " +
      duckdb::Value(value).ToSQLString() + ")");
  };
  run("SET gpu_execution=true");
  run("SET enable_duckdb_fallback=true");
  run("SET parquet_metadata_cache=true");
  run("SET validate_external_file_cache='NO_VALIDATION'");
  auto runtime          = sirius::test::get_registered_sirius_context(con);
  auto const& config    = runtime->get_config().get_scan_manager_config();
  auto counters         = runtime->physical_counters();
  counters->track_units = true;
  std::atomic<unsigned> decoded{0};
  counters->parquet_filter_for_testing = [&](bool, uint64_t rows) {
    if (rows) ++decoded;
  };
  auto clear =
    std::shared_ptr<void>(nullptr, [&](void*) { counters->parquet_filter_for_testing = {}; });
  auto success = [&] {
    auto before = runtime->get_transparent_execution_stats();
    auto reads  = decoded.load();
    CHECK(ducklake_result_bag(*run(sql)) == expected);
    auto after = runtime->get_transparent_execution_stats();
    CHECK(after.executions == before.executions + 1);
    CHECK(after.fallbacks == before.fallbacks);
    CHECK(after.runtime_fallbacks == before.runtime_fallbacks);
    CHECK(decoded.load() > reads);
  };
  std::unique_ptr<cucascade::io::datasource> previous_scope;
  if (warm) {
    endpoint(healthy.endpoint());
    success();
    previous_scope = cache_remote_delete(con, *runtime, uri, bytes);
    auto gets      = healthy.get_count();
    REQUIRE(gets > 0);
    success();
    CHECK(healthy.get_count() == gets);  // The delete bytes really were cached.
  }
  endpoint(failing.endpoint());
  if (fault_kind >= 6) {
    auto provider = std::make_shared<sirius::scan_manager::test::test_reservation_provider>();
    provider->grant_bytes = 128 * 1024 * 1024;
    std::atomic<unsigned> allocation_failures{0};
    if (fault_kind == 7)
      provider->seen->before_allocate = [&] {
        // Leave admission allocations alone; fail only after remote I/O began.
        if (failing.get_count() > 0) {
          ++allocation_failures;
          throw std::bad_alloc();
        }
      };
    counters->preparation_provider_for_testing = provider;
    auto clear_provider                        = std::shared_ptr<void>(nullptr, [&](void*) {
      counters->preparation_provider_for_testing.reset();
      provider->seen->before_allocate = {};
    });
    auto before                                = runtime->get_transparent_execution_stats();
    auto reads                                 = decoded.load();
    duckdb::unique_ptr<duckdb::QueryResult> failed;
    if (fault_kind == 6) {
      auto active = std::async(std::launch::async, [&] { return con.Query(sql); });
      cucascade::test::response_release_guard unblock{response};
      REQUIRE(response->wait_for_holds(1, std::chrono::seconds(15)));
      CHECK(provider->outstanding() > 0);
      con.Interrupt();
      CHECK(active.wait_for(std::chrono::milliseconds(100)) == std::future_status::timeout);
      response->release_all();
      REQUIRE(active.wait_for(std::chrono::seconds(15)) == std::future_status::ready);
      failed = active.get();
      REQUIRE(failed);
      REQUIRE(failed->HasError());
      CHECK(failed->GetErrorType() == duckdb::ExceptionType::INTERRUPT);
    } else {
      failed = con.Query(sql);
      REQUIRE(failed);
      REQUIRE(failed->HasError());
      CHECK(allocation_failures.load() > 0);
      CHECK(failed->GetError().find("preparation backing allocation failed") != std::string::npos);
      CHECK(failed->GetError().find("S3 CPU fallback is not supported") != std::string::npos);
    }
    auto after = runtime->get_transparent_execution_stats();
    if (fault_kind == 7) {
      auto cause = static_cast<size_t>(sirius::transparent::late_failure_cause::resource);
      CHECK(after.late_failures[cause] == before.late_failures[cause] + 1);
      CHECK(after.late_replays[cause] == before.late_replays[cause]);
    }
    CHECK(failing.get_count() > 0);
    CHECK(decoded.load() == reads);
    CHECK(after.fallbacks == before.fallbacks);
    CHECK(after.runtime_fallbacks == before.runtime_fallbacks);
    failed.reset();
    CHECK(provider->outstanding() == 0);
    CHECK(provider->seen->allocated_bytes == 0);
  } else if (puffin && fault_kind < 2) {
    // The shared Puffin contract permits unversioned opens, but never reuses
    // their bytes across opens. Parquet auxiliary reads require strong identity.
    auto cached = cache_remote_delete(con, *runtime, uri, bytes);
    for (int repeat = 0; repeat < 2; ++repeat) {
      auto reads = failing.get_count();
      auto opens = failing.head_count();
      success();
      CHECK(failing.get_count() > reads);
      CHECK(failing.head_count() > opens);
    }
    for (auto const& request : failing.get_requests())
      CHECK(request.if_match.empty());
  } else {
    auto before = runtime->get_transparent_execution_stats();
    auto reads  = decoded.load();
    auto failed = con.Query(sql);
    REQUIRE(failed);
    REQUIRE(failed->HasError());
    CHECK(failed->GetError().find("S3 CPU fallback is not supported") != std::string::npos);
    if (fault_kind >= 4) CHECK(failed->GetError().find("exhausted retries") != std::string::npos);
    auto after = runtime->get_transparent_execution_stats();
    CHECK(after.fallbacks == before.fallbacks);
    CHECK(after.runtime_fallbacks == before.runtime_fallbacks);
    CHECK(decoded.load() == reads);
    CHECK(failing.head_count() > 0);
    if (fault_kind < 2)
      CHECK(failing.get_count() == 0);
    else {
      auto attempts = fault_kind == 4 ? config.rest.max_auth_retry_attempts
                                      : (fault_kind == 5 ? config.rest.max_retry_attempts : 1);
      REQUIRE(failing.get_count() == attempts);
      for (auto const& request : failing.get_requests())
        CHECK(request.if_match == std::vector<std::string>{"\"opened\""});
    }
  }
  endpoint(healthy.endpoint());
  success();
  previous_scope.reset();
  success();  // A fresh query remains valid after the old datasource owner is gone.
}

TEST_CASE("DuckLake remote Puffin byte source retains snapshots and charged ownership",
          "[isolated_context][ducklake][ducklake_remote_transport][s3]")
{
  namespace dl = sirius::ducklake;
  namespace sm = sirius::scan_manager;
  {
    // Fixture generation only needs DuckDB, not a GPU runtime.
    sirius::util::env_guard disabled("SIRIUS_DISABLE", std::string("1"));
    if (sirius::test::s3::skip_or_fail_unless(sirius::test::ensure_s3_test_env(),
                                              "S3 test backend is not configured"))
      return;
  }
  sirius::test::scoped_temp_directory disk;
  sirius::test::shared_test_env env(remote_delete_cache_config(disk.directory));
  auto con = env.make_connection();
  REQUIRE(load_ducklake(con));
  auto run = [&](std::string const& sql) { return query(con, sql); };
  run("SET gpu_execution=false");
  attach_lake(con, disk.directory);
  run("CALL lake.set_option('data_inlining_row_limit',0)");
  run("CALL lake.set_option('write_deletion_vectors',true)");
  run("CREATE TABLE lake.q AS SELECT i::BIGINT id FROM range(128) r(i)");
  run("DELETE FROM lake.q WHERE id=1");
  auto snapshot = run("SELECT max(snapshot_id) FROM ducklake_snapshots('lake')")
                    ->Collection()
                    .GetValue(0, 0)
                    .GetValue<uint64_t>();
  run("DELETE FROM lake.q WHERE id=2");
  auto files = run("SELECT data_file,delete_file FROM ducklake_list_files('lake','q')");
  REQUIRE(files->RowCount() == 1);
  std::ifstream input(files->Collection().GetValue(1, 0).ToString(), std::ios::binary);
  REQUIRE(input.good());
  std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(input)), {});
  REQUIRE(sirius::test::put_s3_test_object("ducklake-phase3/component.puffin", bytes));
  dl::unit_descriptor unit;
  unit.kind        = dl::unit_kind::parquet;
  unit.input.path  = files->Collection().GetValue(0, 0).ToString();
  unit.deletes     = dl::delete_format::puffin;
  unit.delete_path = "s3://" + sirius::test::s3::require_env("SIRIUS_TEST_S3_BUCKET") +
                     "/ducklake-phase3/component.puffin";
  unit.delete_file_size = bytes.size();
  auto runtime          = con.context->registered_state->Get<duckdb::SiriusContext>("sirius_state");
  REQUIRE(runtime);
  cucascade::io::object_store_config config;
  config.endpoint   = sirius::test::s3::require_env("SIRIUS_TEST_S3_ENDPOINT");
  config.region     = sirius::test::s3::env_or("SIRIUS_TEST_S3_REGION", "us-east-1");
  config.access_key = sirius::test::s3::require_env("SIRIUS_TEST_S3_ACCESS_KEY");
  config.secret_key = sirius::test::s3::require_env("SIRIUS_TEST_S3_SECRET_KEY");
  runtime->get_scan_manager().install_s3_config(unit.delete_path, config);
  auto context = runtime->get_scan_manager().create_datasource(unit.delete_path)->io_ctx();
  REQUIRE(context->type() == cucascade::io::io_context_type::restful);
  std::vector<std::pair<std::string, size_t>> fault_offsets;
  for (bool current : {false, true}) {
    unit.bound_snapshot = snapshot + (current ? 1 : 0);
    sm::test::test_reservation_provider provider;
    provider.grant_bytes = 128 * 1024 * 1024;
    sm::preparation_ledger ledger(provider);
    auto envelope = dl::parquet_unit_envelope(unit, dl::dependency_scope::s3_deletes);
    std::array scans{sm::scan_envelope{{envelope}, {}, true}};
    REQUIRE(ledger.admit(scans, sm::memory_space_id(cucascade::memory::Tier::HOST, 0)).deferred);
    auto permit = ledger.try_acquire_unit({1, 1}, envelope.retained);
    REQUIRE(permit);
    auto allocator = ledger.allocator(permit);
    if (!current) {
      snapshot_visibility_puffin_source local(files->Collection().GetValue(1, 0).ToString());
      auto metadata = sirius::op::scan::read_puffin_metadata(local, allocator);
      REQUIRE(metadata->blob_count() == 2);
      REQUIRE(metadata->property(0, "ducklake-snapshot-id").value_or("") ==
              std::to_string(snapshot));
      REQUIRE(metadata->property(1, "ducklake-snapshot-id").value_or("") ==
              std::to_string(snapshot + 1));
      REQUIRE(metadata->blob(0).offset < metadata->blob(1).offset);
      fault_offsets = {{"magic", 0},
                       {"footer tail", bytes.size() - 12},
                       {"footer magic", static_cast<size_t>(metadata->footer_start())},
                       {"footer JSON", static_cast<size_t>(metadata->footer_start() + 4)},
                       {"first blob", static_cast<size_t>(metadata->blob(0).offset)},
                       {"later blob", static_cast<size_t>(metadata->blob(1).offset)}};
    }
    auto visibility = dl::read_dv_visibility(unit,
                                             128,
                                             "component-data-identity",
                                             [context](std::string_view) { return context; },
                                             allocator,
                                             {});
    REQUIRE(visibility->positions.size() == (current ? 2 : 1));
    CHECK(visibility->positions[0] == 1);
    if (current) CHECK(visibility->positions[1] == 2);
    CHECK_FALSE(visibility->delete_identity.empty());
    ledger.retire_unit({1, 1});
    permit.release();
    CHECK(provider.seen->allocated_bytes > 0);
    visibility.reset();
    CHECK(provider.seen->allocated_bytes == 0);
  }
  auto expected = ducklake_result_bag(*run("SELECT id FROM lake.q"));
  REQUIRE(expected.size() == 126);
  run("UPDATE __ducklake_metadata_lake.ducklake_delete_file SET path=" +
      duckdb::Value(unit.delete_path).ToSQLString() +
      ", path_is_relative=false WHERE end_snapshot IS NULL");
  run("SET gpu_execution=true");
  run("SET enable_duckdb_fallback=true");
  auto endpoint = [&](std::string const& value) {
    run("CREATE OR REPLACE SECRET range_fault (TYPE SIRIUS_S3, SCOPE " +
        duckdb::Value(unit.delete_path).ToSQLString() +
        ", KEY_ID 'access', SECRET 'secret', REGION 'us-east-1', ENDPOINT " +
        duckdb::Value(value).ToSQLString() + ")");
  };
  cucascade::test::range_fault_policy good;
  good.successful_head_etag = good.successful_get_etag = "\"range-version\"";
  cucascade::test::loopback_range_server healthy(bytes, good);
  auto counters         = runtime->physical_counters();
  counters->track_units = true;
  std::atomic<unsigned> decoded{0};
  counters->parquet_filter_for_testing = [&](bool, uint64_t rows) {
    if (rows) ++decoded;
  };
  auto clear =
    std::shared_ptr<void>(nullptr, [&](void*) { counters->parquet_filter_for_testing = {}; });
  for (auto const& [site, offset] : fault_offsets) {
    for (bool truncated : {false, true}) {
      CAPTURE(site, offset, truncated);
      sirius::test::offset_fault_server failing(bytes, good.successful_get_etag, offset, truncated);
      endpoint(failing.endpoint());
      auto failed_context = sirius::io::s3::capture_duckdb_s3_context(
        *con.context,
        runtime->get_scan_manager(),
        unit.delete_path,
        runtime->get_config().get_scan_manager_config().object_store);
      sm::test::test_reservation_provider provider;
      provider.grant_bytes = 128 * 1024 * 1024;
      sm::preparation_ledger ledger(provider);
      auto envelope = dl::parquet_unit_envelope(unit, dl::dependency_scope::s3_deletes);
      std::array scans{sm::scan_envelope{{envelope}, {}, true}};
      REQUIRE(ledger.admit(scans, sm::memory_space_id(cucascade::memory::Tier::HOST, 0)).deferred);
      auto permit = ledger.try_acquire_unit({1, 1}, envelope.retained);
      REQUIRE(permit);
      auto allocator = ledger.allocator(permit);
      // A failure in a future blob must also invalidate this older view.
      unit.bound_snapshot = snapshot;
      std::shared_ptr<dl::dv_visibility const> visibility;
      try {
        visibility =
          dl::read_dv_visibility(unit,
                                 128,
                                 "component-data-identity",
                                 [failed_context](std::string_view) { return failed_context; },
                                 allocator,
                                 {});
        FAIL("remote range failure was accepted");
      } catch (sirius::transparent::classified_execution_error const& error) {
        CHECK(error.cause == sirius::transparent::late_failure_cause::reader_io);
        if (!truncated) CHECK(std::string(error.what()).find("HTTP 400") != std::string::npos);
      }
      CHECK_FALSE(visibility);
      CHECK(provider.seen->allocated_bytes == 0);
      ledger.retire_unit({1, 1});
      permit.release();
      CHECK(provider.seen->allocated_bytes == 0);
      auto direct_hits = failing.targeted_reads();
      REQUIRE(direct_hits > 0);
      auto before = runtime->get_transparent_execution_stats();
      auto reads  = decoded.load();
      auto failed = con.Query("SELECT id FROM lake.q");
      REQUIRE(failed);
      REQUIRE(failed->HasError());
      CHECK(failed->GetError().find("S3 CPU fallback is not supported") != std::string::npos);
      CHECK(decoded.load() == reads);
      auto after = runtime->get_transparent_execution_stats();
      CHECK(after.fallbacks == before.fallbacks);
      CHECK(after.runtime_fallbacks == before.runtime_fallbacks);
      CHECK(failing.targeted_reads() >
            direct_hits);  // SQL must independently reach the same fault site.
      CHECK(failing.all_gets_matched_etag());
      if (!truncated) CHECK(failed->GetError().find("HTTP 400") != std::string::npos);
      endpoint(healthy.endpoint());
      CHECK(ducklake_result_bag(*run("SELECT id FROM lake.q")) == expected);
      CHECK(decoded.load() > reads);
      auto recovered = runtime->get_transparent_execution_stats();
      CHECK(recovered.executions == after.executions + 1);
      CHECK(recovered.fallbacks == after.fallbacks);
      CHECK(recovered.runtime_fallbacks == after.runtime_fallbacks);
    }
  }
  // Cached physical bytes do not bypass validation of future snapshot payloads.
  auto crc_offsets =
    snapshot_visibility_dv_crc_offsets(files->Collection().GetValue(1, 0).ToString());
  REQUIRE(crc_offsets.size() == 2);
  auto observed = std::make_shared<observed_host_reservations>(runtime->get_memory_manager());
  counters->preparation_provider_for_testing = observed;
  auto clear_provider                        = std::shared_ptr<void>(
    nullptr, [&](void*) { counters->preparation_provider_for_testing.reset(); });
  for (auto const& [blob_snapshot, crc_offset] : crc_offsets) {
    for (bool warm : {false, true}) {
      CAPTURE(blob_snapshot, warm);
      auto corrupt = bytes;
      REQUIRE(crc_offset < corrupt.size());
      corrupt[crc_offset] ^= 1;
      cucascade::test::loopback_range_server failing(corrupt, good);
      endpoint(failing.endpoint());
      auto cached = warm ? cache_remote_delete(con, *runtime, unit.delete_path, corrupt) : nullptr;
      auto gets   = failing.get_count();
      if (warm) REQUIRE(gets > 0);
      for (auto const& sql :
           {"SELECT id FROM lake.q AT (VERSION => " + std::to_string(snapshot) + ")",
            std::string("SELECT id FROM lake.q")}) {
        auto before = runtime->get_transparent_execution_stats();
        auto reads  = decoded.load();
        auto failed = con.Query(sql);
        REQUIRE(failed);
        REQUIRE(failed->HasError());
        INFO(failed->GetError());
        CHECK(failed->GetError().find("CRC-32 mismatch") != std::string::npos);
        CHECK(failed->GetError().find("S3 CPU fallback is not supported") != std::string::npos);
        CHECK(decoded.load() == reads);
        auto after = runtime->get_transparent_execution_stats();
        CHECK(after.fallbacks == before.fallbacks);
        CHECK(after.runtime_fallbacks == before.runtime_fallbacks);
        failed.reset();
        std::lock_guard lock(observed->seen->mutex);
        CHECK(observed->seen->live == 0);
        CHECK(observed->seen->grants == 0);
      }
      if (warm)
        CHECK(failing.get_count() == gets);
      else
        CHECK(failing.get_count() > 0);
      endpoint(healthy.endpoint());
      CHECK(ducklake_result_bag(*run("SELECT id FROM lake.q")) == expected);
    }
  }
  run("DROP SECRET range_fault");
  auto identity_path = unit.delete_path + ".identity";
  REQUIRE(sirius::test::put_s3_test_object("ducklake-phase3/component.puffin.identity", bytes));
  auto opened = sirius::op::scan::open_puffin_source(
    identity_path, [context](std::string_view) { return context; }, {});
  runtime->get_config().set_object_store_config(config);
  run("SET gpu_execution=true");
  duckdb::OpenFileInfo file(identity_path);
  file.extended_info = duckdb::make_shared_ptr<duckdb::ExtendedOpenFileInfo>();
  file.extended_info->options["sirius_require_strong_identity"] = duckdb::Value::BOOLEAN(true);
  auto native = duckdb::FileSystem::GetFileSystem(*con.context)
                  .OpenFile(file, duckdb::FileFlags::FILE_FLAGS_READ);
  bytes[0] ^= 1;
  REQUIRE(sirius::test::put_s3_test_object("ducklake-phase3/component.puffin.identity", bytes));
  std::array<std::byte, 4> prefix;
  REQUIRE(opened.source->seek(0));
  CHECK_THROWS_AS(opened.source->read(prefix), cucascade::io::object_changed_error);
  CHECK_THROWS_AS(native->Read(prefix.data(), prefix.size(), 0),
                  cucascade::io::object_changed_error);
}

TEST_CASE("DuckLake remote deletes publish earlier files under bounded preparation",
          "[.][isolated_context][ducklake_remote_progress]")
{
  if (!sirius::ducklake::bridge_available()) SKIP("DuckLake private binding bridge is disabled");
  auto puffin  = GENERATE(false, true);
  auto permits = GENERATE(1u, 2u);
  CAPTURE(puffin, permits);
  sirius::test::scoped_temp_directory disk;
  auto config = write_lake_config(disk.directory, [&](YAML::Node& config) {
    config["sirius"]["executor"]["scan_manager"]["num_threads"]                         = 2;
    config["sirius"]["executor"]["scan_manager"]["preparation"]["max_inflight_dv_jobs"] = permits;
  });
  sirius::test::shared_test_env env(config);
  auto con = env.make_connection();
  REQUIRE(load_ducklake(con));
  auto run = [&](std::string const& sql) { return query(con, sql); };
  run("SET gpu_execution=false");
  attach_lake(con, disk.directory);
  run("CALL lake.set_option('data_inlining_row_limit',0)");
  run(std::string("CALL lake.set_option('write_deletion_vectors',") +
      (puffin ? "true)" : "false)"));
  run("CREATE TABLE lake.q(id BIGINT)");
  for (int i = 0; i < 3; ++i)
    run("INSERT INTO lake.q SELECT i FROM range(" + std::to_string(i * 100) + "," +
        std::to_string((i + 1) * 100) + ") r(i)");
  run("DELETE FROM lake.q WHERE id%100=0");
  // DuckLake reuses one preparation slot per scan, not one per file.
  std::string sql = permits == 1 ? "SELECT id FROM lake.q"
                                 : "SELECT id FROM lake.q UNION ALL SELECT id FROM lake.q";
  auto expected   = ducklake_result_bag(*run(sql));
  REQUIRE(expected.size() == 297 * permits);
  auto files = run("SELECT data_file,delete_file FROM ducklake_list_files('lake','q')");
  REQUIRE(files->RowCount() == 3);
  std::array<std::shared_ptr<cucascade::test::range_response_control>, 3> controls;
  std::vector<std::string> data_paths;
  std::vector<std::unique_ptr<cucascade::test::loopback_range_server>> servers;
  for (size_t i = 0; i < 3; ++i) {
    data_paths.push_back(files->Collection().GetValue(0, i).ToString());
    auto path = files->Collection().GetValue(1, i).ToString();
    std::ifstream input(path, std::ios::binary);
    REQUIRE(input.good());
    std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(input)), {});
    cucascade::test::range_fault_policy policy;
    policy.successful_head_etag = policy.successful_get_etag = "\"progress\"";
    controls[i]                  = std::make_shared<cucascade::test::range_response_control>();
    policy.body.control          = controls[i];
    policy.body.hold_after_bytes = 0;
    servers.push_back(std::make_unique<cucascade::test::loopback_range_server>(bytes, policy));
    auto scope = "s3://progress-" + std::to_string(i) + "/";
    run("CREATE SECRET progress_" + std::to_string(i) + " (TYPE SIRIUS_S3, SCOPE " +
        duckdb::Value(scope).ToSQLString() +
        ", KEY_ID 'access', SECRET 'secret', REGION 'us-east-1', ENDPOINT " +
        duckdb::Value(servers.back()->endpoint()).ToSQLString() + ")");
    run("UPDATE __ducklake_metadata_lake.ducklake_delete_file SET path=" +
        duckdb::Value(scope + "delete").ToSQLString() +
        ", path_is_relative=false WHERE ends_with(" + duckdb::Value(path).ToSQLString() +
        ", path)");
    REQUIRE(run("SELECT count(*) FROM __ducklake_metadata_lake.ducklake_delete_file WHERE path=" +
                duckdb::Value(scope + "delete").ToSQLString())
              ->Collection()
              .GetValue(0, 0)
              .GetValue<int64_t>() == 1);
  }
  auto runtime          = sirius::test::get_registered_sirius_context(con);
  auto counters         = runtime->physical_counters();
  counters->track_units = true;
  std::mutex mutex;
  std::set<std::string> decoded;
  counters->parquet_phase_for_testing = [&](std::string const& path, bool footer) {
    if (!footer) {
      std::lock_guard lock(mutex);
      decoded.insert(path);
    }
  };
  auto clear = std::shared_ptr<void>(nullptr, [&](void*) {
    counters->parquet_phase_for_testing = {};
    counters->preparation_provider_for_testing.reset();
    counters->iceberg_statement_route_for_testing = {};
  });
  run("SET gpu_execution=true");
  run("SET enable_duckdb_fallback=true");
  auto refused         = std::make_shared<sirius::scan_manager::test::test_reservation_provider>();
  refused->return_null = true;
  counters->preparation_provider_for_testing = refused;
  bool admission_refused                     = false;
  counters->iceberg_statement_route_for_testing =
    [&](std::string_view route, sirius::scan_manager::admission_decision const& decision) {
      admission_refused = route == "provider_admission_refused" && !decision.deferred;
    };
  auto rejected_before = runtime->get_transparent_execution_stats();
  auto rejected        = con.Query(sql);
  REQUIRE(rejected);
  REQUIRE(rejected->HasError());
  INFO(rejected->GetError());
  CHECK(admission_refused);
  CHECK(refused->seen->requests == 1);
  CHECK(refused->seen->allocations == 0);
  CHECK(refused->outstanding() == 0);
  CHECK(rejected->GetError().find("S3 CPU fallback is not supported") != std::string::npos);
  CHECK(runtime->get_transparent_execution_stats().fallbacks == rejected_before.fallbacks);
  CHECK(runtime->get_transparent_execution_stats().runtime_fallbacks ==
        rejected_before.runtime_fallbacks);
  for (auto const& server : servers) {
    CHECK(server->head_count() == 0);
    CHECK(server->get_count() == 0);
  }
  CHECK(decoded.empty());
  auto observed = std::make_shared<observed_host_reservations>(runtime->get_memory_manager());
  counters->preparation_provider_for_testing = observed;
  std::atomic<unsigned> admissions{0};
  counters->iceberg_statement_route_for_testing =
    [&](std::string_view, sirius::scan_manager::admission_decision const& decision) {
      if (!decision.deferred) throw std::runtime_error("remote preparation was refused");
      if (decision.n_permits != permits)
        throw std::runtime_error("unexpected preparation permit count");
      observed->record_admission(decision, 1, permits);
      ++admissions;
    };
  auto hosts =
    runtime->get_memory_manager().get_memory_spaces_for_tier(cucascade::memory::Tier::HOST);
  REQUIRE(hosts.size() == 1);
  auto reserved = hosts[0]->get_total_reserved_memory();
  run("SET enable_duckdb_fallback=false");
  run("SET scan_task_batch_size=1");
  run("SET sirius_test_hold_footer_index=" + std::to_string(UINT64_MAX));
  auto before  = runtime->get_transparent_execution_stats();
  auto active  = std::async(std::launch::async, [&] { return con.Query(sql); });
  auto unblock = std::shared_ptr<void>(nullptr, [&](void*) {
    for (auto const& control : controls)
      control->release_all();
  });
  // Release the first two distinct files as they start reading. Keep the last
  // response in flight until the earlier files have published their batches.
  std::set<size_t> started;
  size_t blocked = controls.size();
  auto deadline  = std::chrono::steady_clock::now() + std::chrono::seconds(15);
  while (started.size() < controls.size() && std::chrono::steady_clock::now() < deadline) {
    for (size_t i = 0; i < controls.size(); ++i) {
      if (started.contains(i) || !controls[i]->wait_for_holds(1, std::chrono::milliseconds(5)))
        continue;
      started.insert(i);
      if (started.size() < controls.size())
        controls[i]->release_all();
      else
        blocked = i;
    }
  }
  REQUIRE(blocked < data_paths.size());
  CHECK(active.wait_for(std::chrono::milliseconds(0)) == std::future_status::timeout);
  bool published =
    runtime->get_scan_manager().wait_for_publication_for_testing(std::chrono::seconds(15));
  {
    std::lock_guard lock(mutex);
    CHECK_FALSE(decoded.empty());
    CHECK_FALSE(decoded.contains(data_paths[blocked]));
  }
  controls[blocked]->release_all();
  REQUIRE(active.wait_for(std::chrono::seconds(15)) == std::future_status::ready);
  auto result = active.get();
  REQUIRE(published);
  CHECK(ducklake_result_bag(*result) == expected);
  result.reset();

  {
    std::lock_guard lock(mutex);
    REQUIRE(decoded.size() == data_paths.size());
    for (auto const& path : data_paths)
      CHECK(decoded.contains(path));
  }
  CHECK(hosts[0]->get_total_reserved_memory() == reserved);
  run("SET sirius_test_hold_footer_index=0");
  CHECK(ducklake_result_bag(*run(sql)) == expected);
  auto after = runtime->get_transparent_execution_stats();
  CHECK(after.executions == before.executions + 2);
  CHECK(after.fallbacks == before.fallbacks);
  CHECK(after.runtime_fallbacks == before.runtime_fallbacks);
  CHECK(admissions.load() == 2);
  CHECK(hosts[0]->get_total_reserved_memory() == reserved);
  std::lock_guard lock(observed->seen->mutex);
  CHECK_FALSE(observed->seen->exceeded);
  CHECK(observed->seen->peak_live > 0);
  CHECK(observed->seen->live == 0);
  CHECK(observed->seen->grants == 0);
}

TEST_CASE("DuckLake file pruning excludes unsupported remote dependencies before admission",
          "[.][ducklake_remote_file_pruning][isolated_context]")
{
  if (!sirius::ducklake::bridge_available()) SKIP("DuckLake private binding bridge is disabled");
  auto puffin      = GENERATE(false, true);
  auto remote_data = GENERATE(false, true);
  CAPTURE(puffin, remote_data);
  sirius::test::scoped_temp_directory disk;
  sirius::test::shared_test_env env(lake_config());
  auto con = env.make_connection();
  REQUIRE(load_ducklake(con));
  auto run = [&](std::string const& sql) { return query(con, sql); };
  run("SET gpu_execution=false");
  attach_lake(con, disk.directory);
  run("CALL lake.set_option('data_inlining_row_limit',0)");
  run(std::string("CALL lake.set_option('write_deletion_vectors',") +
      (puffin ? "true)" : "false)"));
  run("CREATE TABLE lake.q(id BIGINT)");
  run("INSERT INTO lake.q SELECT i FROM range(100) r(i)");
  run("INSERT INTO lake.q SELECT i FROM range(100,200) r(i)");
  run("DELETE FROM lake.q WHERE id IN (1,101)");
  auto low_sql  = "SELECT id FROM lake.q WHERE id>=0 AND id<100";
  auto high_sql = "SELECT id FROM lake.q WHERE id>=100 AND id<200";
  auto low      = ducklake_result_bag(*run(low_sql));
  auto high     = ducklake_result_bag(*run(high_sql));
  REQUIRE(low.size() == 99);
  REQUIRE(high.size() == 99);
  auto files = run("SELECT data_file,delete_file FROM ducklake_list_files('lake','q')");
  REQUIRE(files->RowCount() == 2);
  std::string selected_path, excluded_path;
  for (duckdb::idx_t i = 0; i < files->RowCount(); ++i) {
    auto path = files->Collection().GetValue(0, i).ToString();
    auto minimum =
      run("SELECT min(id) FROM read_parquet(" + duckdb::Value(path).ToSQLString() + ")")
        ->Collection()
        .GetValue(0, 0)
        .GetValue<int64_t>();
    if (minimum == 0)
      selected_path = path;
    else {
      REQUIRE(minimum == 100);
      excluded_path = files->Collection().GetValue(remote_data ? 0 : 1, i).ToString();
    }
  }
  REQUIRE_FALSE(selected_path.empty());
  REQUIRE_FALSE(excluded_path.empty());
  std::ifstream input(excluded_path, std::ios::binary);
  REQUIRE(input.good());
  std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(input)), {});
  cucascade::test::loopback_range_server server(bytes);
  auto uri      = server.endpoint() + "/unsupported";
  auto metadata = std::string("__ducklake_metadata_lake.") +
                  (remote_data ? "ducklake_data_file" : "ducklake_delete_file");
  run("UPDATE " + metadata + " SET path=" + duckdb::Value(uri).ToSQLString() +
      ", path_is_relative=false WHERE ends_with(" + duckdb::Value(excluded_path).ToSQLString() +
      ",path)");
  REQUIRE(
    run("SELECT count(*) FROM " + metadata + " WHERE path=" + duckdb::Value(uri).ToSQLString())
      ->Collection()
      .GetValue(0, 0)
      .GetValue<int64_t>() == 1);
  auto runtime          = sirius::test::get_registered_sirius_context(con);
  auto counters         = runtime->physical_counters();
  counters->track_units = true;
  std::atomic<unsigned> footers{0}, decoded{0};
  counters->parquet_phase_for_testing = [&](std::string const&, bool footer) {
    if (footer)
      ++footers;
    else
      ++decoded;
  };
  auto observed = std::make_shared<observed_host_reservations>(runtime->get_memory_manager());
  counters->preparation_provider_for_testing = observed;
  auto clear                                 = std::shared_ptr<void>(nullptr, [&](void*) {
    counters->parquet_phase_for_testing = {};
    counters->preparation_provider_for_testing.reset();
  });
  run("SET enable_duckdb_fallback=false");
  run("SET gpu_execution=true");
  run("PREPARE selected_file AS SELECT id FROM lake.q WHERE id >= $1 AND id < $2");
  for (bool prepared : {false, true}) {
    CAPTURE(prepared);
    auto before = runtime->get_transparent_execution_stats();
    auto reads  = decoded.load();
    CHECK(ducklake_result_bag(*run(prepared ? "EXECUTE selected_file(0,100)" : low_sql)) == low);
    CHECK(decoded.load() > reads);
    auto footer_count = footers.load();
    reads             = decoded.load();
    auto rejected     = con.Query(prepared ? "EXECUTE selected_file(100,200)" : high_sql);
    REQUIRE(rejected);
    REQUIRE(rejected->HasError());
    INFO(rejected->GetError());
    CHECK(rejected->GetError().find(remote_data ? "DuckLake remote data is outside"
                                                : "DuckLake remote deletes are outside") !=
          std::string::npos);
    CHECK(footers.load() == footer_count);
    CHECK(decoded.load() == reads);
    rejected.reset();
    CHECK(ducklake_result_bag(*run(prepared ? "EXECUTE selected_file(0,100)" : low_sql)) == low);
    auto after = runtime->get_transparent_execution_stats();
    CHECK(after.fallbacks == before.fallbacks);
    CHECK(after.runtime_fallbacks == before.runtime_fallbacks);
    CHECK(server.head_count() == 0);
    CHECK(server.get_count() == 0);
    std::lock_guard lock(observed->seen->mutex);
    CHECK(observed->seen->live == 0);
    CHECK(observed->seen->grants == 0);
  }
  run("UPDATE " + metadata + " SET path=" + duckdb::Value(excluded_path).ToSQLString() +
      " WHERE path=" + duckdb::Value(uri).ToSQLString());
  CHECK(ducklake_result_bag(*run(high_sql)) == high);
}

TEST_CASE("DuckLake validates invisible Parquet deletes across parallel row group boundaries",
          "[ducklake][ducklake_parallel_deletes][isolated_context]")
{
  if (!sirius::ducklake::bridge_available()) SKIP("DuckLake private binding bridge is disabled");
  sirius::test::scoped_temp_directory disk;
  sirius::test::shared_test_env env(lake_config());
  auto con = env.make_connection();
  REQUIRE(load_ducklake(con));
  auto run = [&](std::string const& sql) { return query(con, sql); };
  run("SET gpu_execution=false");
  run("SET threads=1");
  attach_lake(con, disk.directory);
  run("CALL lake.set_option('data_inlining_row_limit',0)");
  run("CALL lake.set_option('write_deletion_vectors',false)");
  run("CREATE TABLE lake.q AS SELECT i::BIGINT id FROM range(16384) t(i)");
  run("DELETE FROM lake.q WHERE id % 2 = 0");
  auto files = run("SELECT delete_file FROM ducklake_list_files('lake','q')");
  REQUIRE(files->RowCount() == 1);
  auto source = files->Collection().GetValue(0, 0).ToSQLString();
  for (bool corrupt : {false, true}) {
    CAPTURE(corrupt);
    run("SET gpu_execution=false");
    run("SET threads=1");
    auto path =
      std::filesystem::path(disk.directory) / (corrupt ? "duplicate.parquet" : "valid.parquet");
    auto dest = duckdb::Value(path.string()).ToSQLString();
    // Row 2048 starts the second group. A future delete must still participate
    // in global order validation, even though it does not hide row 4096.
    std::string pos = corrupt ? "CASE WHEN pos=4096 THEN 4094 ELSE pos END" : "pos";
    run("COPY (SELECT file_path," + pos +
        " AS pos, CASE WHEN pos=4096 THEN 999999 ELSE 0 END"
        "::BIGINT AS _DuckLake_Internal_Snapshot_Id FROM read_parquet(" +
        source + ") t ORDER BY t.pos) TO " + dest + " (FORMAT PARQUET, ROW_GROUP_SIZE 2048)");
    REQUIRE(run("SELECT count(DISTINCT row_group_id) FROM parquet_metadata(" + dest + ")")
              ->Collection()
              .GetValue(0, 0)
              .GetValue<int64_t>() == 4);
    run("UPDATE __ducklake_metadata_lake.ducklake_delete_file SET path=" + dest +
        ",path_is_relative=false,footer_size=NULL,file_size_bytes=" +
        std::to_string(std::filesystem::file_size(path)));
    run("SET threads=4");
    run("SET enable_duckdb_fallback=false");
    run("SET gpu_execution=true");
    auto result = con.Query("SELECT sum(id) FROM lake.q");
    REQUIRE(result);
    if (corrupt) {
      REQUIRE(result->HasError());
      CHECK(result->GetError().find("strictly increasing") != std::string::npos);
    } else {
      INFO((result->HasError() ? result->GetError() : ""));
      REQUIRE_FALSE(result->HasError());
      CHECK(result->Fetch()->GetValue(0, 0).GetValue<int64_t>() == 8192LL * 8192 + 4096);
    }
  }
}
