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

#include "op/scan/iceberg_dv_preparation.hpp"
#include "op/scan/iceberg_gpu_ingestible.hpp"
#include "op/sirius_physical_table_scan.hpp"
#include "planner/connector_registry.hpp"
#include "planner/sirius_physical_plan_generator.hpp"
#include "scan_manager/preparation_test_support.hpp"
#include "utils/parquet_fixture_utils.hpp"
#include "utils/sirius_test_env.hpp"
#include "utils/transparent_execution_test_utils.hpp"

#include <catch.hpp>
#include <duckdb/common/string_util.hpp>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
using namespace sirius::scan_manager;
using sirius::op::scan::scan_contract_id;
using sirius::op::scan::scan_verdict_declined;
using sirius::op::scan::verdict_reason;
using sirius::planner::connector;
using sirius::planner::testing::scoped_connector_overrides;
namespace fs = std::filesystem;

constexpr char probe_key[]     = "connector_admission_probe";
constexpr char remote_puffin[] = "s3://connector-admission.invalid/sidecar.puffin";
enum class fault { none, prepare, bind, malformed };

struct request_observation {
  unsigned prepares = 0, binds = 0, rollbacks = 0, lowerings = 0;
  unsigned bind_order = 0, rollback_order = 0, lower_order = 0;
  bool lower_saw_admission = false;
  charged_block arena;
  std::shared_ptr<preparation_admission> admission;
};

struct admission_probe : duckdb::ClientContextState {
  fault injected = fault::none;
  std::map<scan_contract_id, request_observation> requests;
  unsigned clock = 0, observations = 0, observer_order = 0;
  std::string route;
  admission_decision decision;
  scan_contract_id iceberg_contract = 0;
  unsigned iceberg_lowerings        = 0;
  bool iceberg_deferred = false, shared_admission = false, eager_data_present = false;
  std::vector<std::pair<scan_contract_id, bool>> dv_routes;

  void release()
  {
    for (auto& [id, request] : requests) {
      request.arena.release();
      request.admission.reset();
    }
  }
};

struct admission_provider : reservation_provider {
  test::test_reservation_provider backing;
  bool fail_before_prepare           = false;
  mutable unsigned granularity_calls = 0;

  admission_provider()
  {
    backing.grant_bytes = 1024 * 1024;
    backing.quantum     = 8;
  }

  uint64_t allocation_granularity(memory_space_id space) const override
  {
    ++granularity_calls;
    if (fail_before_prepare)
      throw preparation_resource_error("injected pre-admit resource fault", false);
    return backing.allocation_granularity(space);
  }

  std::optional<reservation_grant> request(memory_space_id space, uint64_t bytes) override
  {
    return backing.request(space, bytes);
  }
};

auto probe_for(duckdb::ClientContext& context)
{
  auto probe = context.registered_state->Get<admission_probe>(probe_key);
  if (!probe) throw std::logic_error("connector admission probe is missing");
  return probe;
}

std::optional<preparation_request> collect_request(sirius::op::sirius_physical_table_scan& scan,
                                                   sirius::operator_params const&,
                                                   duckdb::ClientContext& context)
{
  auto probe        = probe_for(context);
  auto* observation = &probe->requests[scan.contract_id];
  preparation_request request;
  request.envelope.qualified       = true;
  request.envelope.minimum_permits = 1;
  request.envelope.units.push_back({{64}, 8});
  request.prepare = [probe, observation, contract = scan.contract_id](preparation_ledger& ledger) {
    ++observation->prepares;
    auto permit = ledger.try_acquire_unit({contract, 0}, 8);
    if (!permit) throw std::runtime_error("connector planning permit missing");
    observation->arena = ledger.allocator(permit).allocate_retained(8);
    if (probe->injected == fault::prepare)
      throw preparation_resource_error("injected connector prepare fault", false);
  };
  request.bind = [probe, observation](std::shared_ptr<preparation_admission> admission) {
    ++observation->binds;
    observation->bind_order = ++probe->clock;
    observation->admission  = std::move(admission);
    if (probe->injected == fault::bind)
      throw preparation_resource_error("injected connector bind fault", false);
  };
  if (probe->injected != fault::malformed) {
    request.rollback = [probe, observation]() noexcept {
      ++observation->rollbacks;
      observation->rollback_order = ++probe->clock;
      observation->arena.release();
      observation->admission.reset();
    };
  }
  return request;
}

std::optional<preparation_request> observe_iceberg(sirius::op::sirius_physical_table_scan& scan,
                                                   sirius::operator_params const&,
                                                   duckdb::ClientContext& context)
{
  probe_for(context)->iceberg_contract = scan.contract_id;
  return std::nullopt;
}

duckdb::unique_ptr<sirius::op::sirius_physical_operator> lower_parquet(
  sirius::op::sirius_physical_table_scan& scan,
  sirius::operator_params const& params,
  duckdb::ClientContext& context,
  sirius::op::scan::dynamic_filter_apply_mode mode)
{
  auto probe        = probe_for(context);
  auto& observation = probe->requests.at(scan.contract_id);
  ++observation.lowerings;
  observation.lower_order = ++probe->clock;
  observation.lower_saw_admission =
    observation.admission && observation.admission->decision.deferred;
  return sirius::planner::lower_parquet_scan(scan, params, context, mode);
}

duckdb::unique_ptr<sirius::op::sirius_physical_operator> lower_iceberg(
  sirius::op::sirius_physical_table_scan& scan,
  sirius::operator_params const& params,
  duckdb::ClientContext& context,
  sirius::op::scan::dynamic_filter_apply_mode mode)
{
  auto probe = probe_for(context);
  ++probe->iceberg_lowerings;
  auto const& info = scan.prepared_iceberg_info;
  if (info) {
    probe->iceberg_deferred   = bool(info->deferred);
    probe->eager_data_present = bool(info->delete_data);
    if (info->deferred && !probe->requests.empty()) {
      probe->shared_admission = true;
      for (auto const& [id, request] : probe->requests)
        probe->shared_admission &= request.admission == info->deferred->admission();
    }
  }
  return sirius::planner::lower_iceberg_scan(scan, params, context, mode);
}

fs::path inventory_fixture(char const* name)
{
  static sirius::test::scratch_dir generated("connector_admission_inventory");
  static bool ready = [] {
    auto quote = [](std::string const& value) {
      return "'" + duckdb::StringUtil::Replace(value, "'", "'\\''") + "'";
    };
    auto script =
      fs::path(SIRIUS_PROJECT_ROOT) / "test/cpp/integration/data/generate_inventory_fixtures.py";
    auto command = "python3 -B " + quote(script.string()) + " " + quote(generated.path().string());
    REQUIRE(std::system(command.c_str()) == 0);
    return true;
  }();
  (void)ready;
  return generated.path() / name;
}

std::string iceberg_scan(char const* fixture)
{
  auto metadata = inventory_fixture(fixture) / "metadata/v1.metadata.json";
  std::ifstream input(metadata);
  REQUIRE(input.good());
  std::string text{std::istreambuf_iterator<char>(input), {}};
  auto key = text.find("\"current-snapshot-id\"");
  REQUIRE(key != std::string::npos);
  auto colon = text.find(':', key);
  REQUIRE(colon != std::string::npos);
  auto snapshot = std::stoll(text.substr(colon + 1));
  return "iceberg_scan(" + sirius::test::sql_literal(metadata.string()) +
         ", snapshot_from_id=" + std::to_string(snapshot) + ")";
}

struct test_generator : sirius::planner::sirius_physical_plan_generator {
  using sirius_physical_plan_generator::create_plan;
  using sirius_physical_plan_generator::sirius_physical_plan_generator;
};

struct admission_fixture {
  sirius::test::scratch_dir files{"connector_admission"};
  std::unique_ptr<duckdb::DuckDB> db =
    sirius::test::open_sirius_db(nullptr, sirius::test::integration_config_path());
  duckdb::Connection con{*db};
  duckdb::shared_ptr<duckdb::SiriusContext> context =
    sirius::test::get_registered_sirius_context(con);
  duckdb::shared_ptr<admission_probe> probe    = duckdb::make_shared_ptr<admission_probe>();
  std::shared_ptr<admission_provider> provider = std::make_shared<admission_provider>();
  std::unique_ptr<scoped_connector_overrides> overrides;
  duckdb::unique_ptr<sirius::op::sirius_physical_operator> plan;

  admission_fixture()
  {
    REQUIRE(sirius::planner::testing::connector_overrides_linked());
    run_ok("SET gpu_execution=false");
    run_ok("LOAD iceberg");
    run_ok("COPY (SELECT 42 AS id) TO " + files.file_literal("a.parquet") + " (FORMAT PARQUET)");
    run_ok("COPY (SELECT 43 AS id) TO " + files.file_literal("b.parquet") + " (FORMAT PARQUET)");
    con.context->registered_state->Insert(probe_key, probe);
    auto counters                                 = context->physical_counters();
    counters->track_units                         = true;
    counters->preparation_provider_for_testing    = provider;
    counters->iceberg_statement_route_for_testing = [state = probe](auto reason,
                                                                    auto const& decision) {
      ++state->observations;
      state->observer_order = ++state->clock;
      state->route          = reason;
      state->decision       = decision;
    };
    counters->iceberg_preparation_route_for_testing = [state = probe](auto id, bool deferred) {
      state->dv_routes.emplace_back(id, deferred);
    };
    std::vector<connector> entries;
    for (auto const& registered : sirius::planner::registered_connectors()) {
      if (registered.function_name == "parquet_scan" ||
          registered.function_name == "read_parquet") {
        auto entry    = registered;
        entry.prepare = collect_request;
        entry.lower   = lower_parquet;
        entries.push_back(std::move(entry));
      } else if (registered.function_name == "iceberg_scan") {
        auto entry    = registered;
        entry.prepare = observe_iceberg;
        entry.lower   = lower_iceberg;
        entries.push_back(std::move(entry));
      }
    }
    REQUIRE(entries.size() == 3);
    overrides = std::make_unique<scoped_connector_overrides>(*db->instance, std::move(entries));
    con.BeginTransaction();
  }

  ~admission_fixture()
  {
    plan.reset();
    probe->release();
    auto counters = context->physical_counters();
    counters->preparation_provider_for_testing.reset();
    counters->iceberg_statement_route_for_testing   = {};
    counters->iceberg_preparation_route_for_testing = {};
    counters->statement_dv_limit_for_testing.reset();
  }

  void run_ok(std::string const& sql)
  {
    auto result = con.Query(sql);
    REQUIRE(result);
    if (result->HasError()) UNSCOPED_INFO(result->GetError());
    REQUIRE_FALSE(result->HasError());
  }

  std::string parquet(char const* name = "a.parquet") const
  {
    return "read_parquet(" + files.file_literal(name) + ")";
  }

  std::string mixed(char const* fixture = "dv_bounded") const
  {
    return "SELECT p.id, i.count FROM " + parquet() + " p CROSS JOIN " + iceberg_scan(fixture) +
           " i";
  }

  std::string two_connectors() const
  {
    return "SELECT a.id, b.id FROM " + parquet() + " a CROSS JOIN " + parquet("b.parquet") + " b";
  }

  void attempt(std::string const& sql)
  {
    auto logical = con.ExtractPlan(sql);
    REQUIRE(logical);
    test_generator generator(*con.context,
                             sirius::planner::scan_contract_provenance{std::nullopt, 1});
    plan = generator.create_plan(std::move(logical));
    REQUIRE(plan);
  }

  auto stats() const { return context->get_transparent_execution_stats(); }
};

struct refusal {
  std::string message;
  std::vector<std::pair<scan_contract_id, verdict_reason>> declined;
};

refusal expect_refusal(admission_fixture& fixture, std::string const& sql)
{
  try {
    fixture.attempt(sql);
  } catch (scan_verdict_declined const& error) {
    return {error.what(), error.declined};
  }
  FAIL("expected scan_verdict_declined");
  return {};
}

void check_refusal(admission_fixture const& fixture,
                   refusal const& error,
                   char const* cause,
                   unsigned prepares = 0,
                   unsigned binds    = 0)
{
  REQUIRE_FALSE(fixture.probe->requests.empty());
  CHECK(error.message.find(cause) != std::string::npos);
  std::map<scan_contract_id, verdict_reason> declined(error.declined.begin(), error.declined.end());
  REQUIRE(error.declined.size() == fixture.probe->requests.size());
  for (auto const& [id, observation] : fixture.probe->requests) {
    REQUIRE(declined.count(id) == 1);
    CHECK(declined.at(id) == verdict_reason::provider_admission_refused);
    CHECK(observation.prepares == prepares);
    CHECK(observation.binds == binds);
    CHECK(observation.rollbacks == 1);
    CHECK(observation.lowerings == 0);
    CHECK_FALSE(observation.admission);
    CHECK(observation.rollback_order < fixture.probe->observer_order);
  }
  CHECK(fixture.probe->observations == 1);
  CHECK(fixture.probe->route == "provider_admission_refused");
  CHECK_FALSE(fixture.probe->decision.deferred);
  CHECK(fixture.provider->backing.outstanding() == 0);
  CHECK(fixture.provider->backing.seen->allocated_bytes.load() == 0);
}

constexpr auto refused_index = static_cast<size_t>(verdict_reason::provider_admission_refused);
}  // namespace

TEST_CASE("Required connector preparation is bound before lowering",
          "[planner][scan_preparation][connector_admission][isolated_context]")
{
  admission_fixture fixture;
  fixture.attempt("SELECT id FROM " + fixture.parquet());
  REQUIRE(fixture.probe->requests.size() == 1);
  auto const& observation = fixture.probe->requests.begin()->second;
  CHECK(observation.prepares == 1);
  CHECK(observation.binds == 1);
  CHECK(observation.rollbacks == 0);
  CHECK(observation.lowerings == 1);
  CHECK(observation.lower_saw_admission);
  CHECK(observation.bind_order < observation.lower_order);
  CHECK(fixture.probe->observations == 1);
  CHECK(fixture.probe->route == "no_deferred_dv_work");
  CHECK(fixture.probe->decision.deferred);
  REQUIRE(observation.admission);
  CHECK(observation.admission->decision.deferred);
  CHECK(fixture.stats().semantic_declines[refused_index] == 0);
}

TEST_CASE("Required HOST refusals cover every originating connector once",
          "[planner][scan_preparation][connector_admission][isolated_context]")
{
  auto denied = GENERATE(true, false);
  admission_fixture fixture;
  fixture.provider->backing.return_null = denied;
  fixture.provider->backing.grant_bytes = denied ? 1024 * 1024 : 1;
  auto before                           = fixture.stats();
  auto error                            = expect_refusal(fixture, fixture.two_connectors());
  REQUIRE(fixture.probe->requests.size() == 2);
  check_refusal(fixture, error, "HOST admission refused");
  CHECK(fixture.stats().semantic_declines[refused_index] ==
        before.semantic_declines[refused_index] + 1);
}

TEST_CASE("Prepare and bind resource faults do not repeat helper rollback",
          "[planner][scan_preparation][connector_admission][isolated_context]")
{
  auto injected = GENERATE(fault::prepare, fault::bind);
  admission_fixture fixture;
  fixture.probe->injected = injected;
  auto before             = fixture.stats();
  auto error              = expect_refusal(fixture, "SELECT id FROM " + fixture.parquet());
  check_refusal(fixture, error, "arena allocation failed", 1, injected == fault::bind ? 1 : 0);
  CHECK(fixture.provider->backing.seen->allocations.load() == 1);
  CHECK(fixture.stats().semantic_declines[refused_index] ==
        before.semantic_declines[refused_index] + 1);
}

TEST_CASE("A resource fault before the helper try still rolls back required requests",
          "[planner][scan_preparation][connector_admission][isolated_context]")
{
  admission_fixture fixture;
  fixture.provider->fail_before_prepare = true;
  auto before                           = fixture.stats();
  auto error = expect_refusal(fixture, "SELECT id FROM " + fixture.parquet());
  check_refusal(fixture, error, "arena allocation failed");
  CHECK(fixture.provider->granularity_calls == 1);
  CHECK(fixture.provider->backing.seen->requests.load() == 0);
  CHECK(fixture.provider->backing.seen->allocations.load() == 0);
  CHECK(fixture.stats().semantic_declines[refused_index] ==
        before.semantic_declines[refused_index] + 1);
}

TEST_CASE("Malformed required requests fail at collection",
          "[planner][scan_preparation][connector_admission][isolated_context]")
{
  admission_fixture fixture;
  fixture.probe->injected = fault::malformed;
  auto before             = fixture.stats();
  try {
    fixture.attempt(fixture.two_connectors());
    FAIL("expected invalid_argument");
  } catch (std::invalid_argument const& error) {
    CHECK(std::string(error.what()) == "incomplete preparation resource request");
  }
  REQUIRE(fixture.probe->requests.size() == 1);
  auto const& observation = fixture.probe->requests.begin()->second;
  CHECK(observation.prepares == 0);
  CHECK(observation.binds == 0);
  CHECK(observation.rollbacks == 0);
  CHECK(observation.lowerings == 0);
  CHECK(fixture.probe->observations == 0);
  CHECK(fixture.provider->granularity_calls == 0);
  CHECK(fixture.provider->backing.seen->requests.load() == 0);
  CHECK(fixture.stats().semantic_declines == before.semantic_declines);
}

TEST_CASE("Mixed DV policy does not bypass required connector admission",
          "[planner][scan_preparation][connector_admission][isolated_context]")
{
  auto mode = GENERATE(0, 1, 2);  // shared admission, denied, DV-only policy exclusion
  admission_fixture fixture;
  fixture.provider->backing.return_null = mode == 1;
  if (mode == 2) fixture.context->physical_counters()->statement_dv_limit_for_testing = 1;
  auto before = fixture.stats();
  auto reads  = sirius::op::scan::iceberg_delete_data_uncached_read_count();
  if (mode == 1) {
    auto error = expect_refusal(fixture, fixture.mixed());
    check_refusal(fixture, error, "HOST admission refused");
    CHECK(fixture.probe->iceberg_lowerings == 0);
    CHECK(fixture.stats().semantic_declines[refused_index] ==
          before.semantic_declines[refused_index] + 1);
  } else {
    fixture.attempt(fixture.mixed());
    REQUIRE(fixture.probe->requests.size() == 1);
    auto const& observation = fixture.probe->requests.begin()->second;
    CHECK(observation.prepares == 1);
    CHECK(observation.binds == 1);
    CHECK(observation.rollbacks == 0);
    CHECK(observation.lower_saw_admission);
    CHECK(fixture.probe->decision.deferred);
    CHECK(fixture.probe->iceberg_lowerings == 1);
    CHECK(fixture.probe->iceberg_deferred == (mode == 0));
    CHECK(fixture.probe->shared_admission == (mode == 0));
    CHECK(fixture.probe->eager_data_present == (mode == 2));
    CHECK(fixture.probe->route == (mode == 0 ? "deferred" : "statement_dv_limit"));
    REQUIRE(fixture.probe->dv_routes.size() == 1);
    CHECK(fixture.probe->dv_routes[0].first == fixture.probe->iceberg_contract);
    CHECK(fixture.probe->dv_routes[0].second == (mode == 0));
  }
  CHECK(sirius::op::scan::iceberg_delete_data_uncached_read_count() == reads + (mode == 2 ? 1 : 0));
}

TEST_CASE("Puffin-primary required refusals still publish the cleanup observation",
          "[planner][scan_preparation][connector_admission][isolated_context]")
{
  admission_fixture fixture;
  auto backend = fixture.context->get_scan_manager().planning_io_backend(remote_puffin);
  REQUIRE(backend.has_value());
  REQUIRE((*backend == sirius::io::io_context_type::restful ||
           *backend == sirius::io::io_context_type::kvikio));
  fixture.provider->backing.return_null = true;
  auto before                           = fixture.stats();
  auto reads = sirius::op::scan::iceberg_delete_data_uncached_read_count();
  auto error = expect_refusal(fixture, fixture.mixed("dv_supported_s3_uri"));
  REQUIRE(error.declined.size() == 1);
  CHECK(error.declined[0].first == fixture.probe->iceberg_contract);
  CHECK(error.declined[0].second == verdict_reason::iceberg_puffin_remote_eager);
  CHECK(error.message.find("remote Puffin requires deferred preparation") != std::string::npos);
  REQUIRE(fixture.probe->requests.size() == 1);
  auto const& observation = fixture.probe->requests.begin()->second;
  CHECK(observation.prepares == 0);
  CHECK(observation.binds == 0);
  CHECK(observation.rollbacks == 1);
  CHECK(observation.lowerings == 0);
  CHECK(observation.rollback_order < fixture.probe->observer_order);
  CHECK(fixture.probe->observations == 1);
  CHECK(fixture.probe->route == "provider_admission_refused");
  CHECK(fixture.probe->iceberg_lowerings == 0);
  auto puffin = static_cast<size_t>(verdict_reason::iceberg_puffin_remote_eager);
  CHECK(fixture.stats().semantic_declines[puffin] == before.semantic_declines[puffin] + 1);
  CHECK(fixture.stats().semantic_declines[refused_index] ==
        before.semantic_declines[refused_index]);
  CHECK(sirius::op::scan::iceberg_delete_data_uncached_read_count() == reads);
  CHECK(fixture.provider->backing.outstanding() == 0);
}

TEST_CASE("Pure Iceberg retains admitted and eager preparation routes",
          "[planner][scan_preparation][connector_admission][isolated_context]")
{
  auto deferred = GENERATE(true, false);
  admission_fixture fixture;
  fixture.provider->backing.return_null = !deferred;
  auto before                           = fixture.stats();
  auto reads = sirius::op::scan::iceberg_delete_data_uncached_read_count();
  fixture.attempt("SELECT count FROM " + iceberg_scan("dv_bounded"));
  CHECK(fixture.probe->requests.empty());
  CHECK(fixture.probe->iceberg_lowerings == 1);
  CHECK(fixture.probe->iceberg_deferred == deferred);
  CHECK(fixture.probe->eager_data_present == !deferred);
  CHECK(fixture.probe->decision.deferred == deferred);
  CHECK(fixture.probe->route == (deferred ? "deferred" : "host_admission"));
  CHECK(fixture.stats().semantic_declines == before.semantic_declines);
  CHECK(sirius::op::scan::iceberg_delete_data_uncached_read_count() == reads + (deferred ? 0 : 1));
}
