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

#include "utils/connector_admission_test_utils.hpp"

#include <catch.hpp>
#include <cucascade/io/io_context.hpp>

#include <typeinfo>

using namespace sirius::test::connector_admission;

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
  auto injected =
    GENERATE(fault::prepare, fault::bind, fault::prepare_bad_alloc, fault::bind_bad_alloc);
  admission_fixture fixture;
  fixture.probe->injected = injected;
  auto before             = fixture.stats();
  auto error              = expect_refusal(fixture, "SELECT id FROM " + fixture.parquet());
  auto bind_failure       = injected == fault::bind || injected == fault::bind_bad_alloc;
  check_refusal(fixture, error, "arena allocation failed", 1, bind_failure ? 1 : 0);
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

TEST_CASE("Unrelated connector faults propagate unchanged and release reserved memory",
          "[planner][scan_preparation][connector_admission][isolated_context]")
{
  auto injected = GENERATE(fault::prepare_unrelated, fault::bind_unrelated);
  admission_fixture fixture;
  fixture.probe->injected = injected;
  auto before             = fixture.stats();
  try {
    fixture.attempt(fixture.two_connectors());
    FAIL("expected the original provider schema error");
  } catch (std::runtime_error const& error) {
    CHECK(typeid(error) == typeid(std::runtime_error));
    CHECK(std::string(error.what()) == "provider schema unavailable");
  }
  REQUIRE(fixture.probe->requests.size() == 2);
  unsigned prepares = 0, binds = 0;
  for (auto const& [id, observation] : fixture.probe->requests) {
    CHECK(observation.rollbacks == 1);
    CHECK(observation.lowerings == 0);
    CHECK_FALSE(observation.arena);
    CHECK_FALSE(observation.admission);
    prepares += observation.prepares;
    binds += observation.binds;
  }
  CHECK(prepares == (injected == fault::prepare_unrelated ? 1 : 2));
  CHECK(binds == (injected == fault::bind_unrelated ? 1 : 0));
  CHECK(fixture.provider->backing.seen->requests.load() == 1);
  CHECK(fixture.provider->backing.seen->allocations.load() == prepares);
  CHECK(fixture.provider->backing.outstanding() == 0);
  CHECK(fixture.provider->backing.seen->allocated_bytes.load() == 0);
  CHECK_FALSE(fixture.plan);
  CHECK(fixture.stats().scan_lowerings.empty());
  CHECK(fixture.stats().semantic_declines[refused_index] ==
        before.semantic_declines[refused_index]);
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
  REQUIRE(*backend == cucascade::io::io_context_type::restful);
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
