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

#include "scan/dict_fsst_fixture.hpp"
#include "utils/sirius_test_env.hpp"
#include "utils/transparent_execution_test_utils.hpp"

#include <catch.hpp>
#include <duckdb.hpp>

#include <memory>
#include <vector>

namespace {

using sirius::test::dict_fsst::query;

struct scope_fixture {
  duckdb::Connection a;
  std::vector<std::unique_ptr<duckdb::Connection>> connections;

  scope_fixture() : a(sirius::test::g_integration_env->make_connection())
  {
    query(a, "RESET GLOBAL gpu_execution");
  }
  ~scope_fixture()
  {
    for (auto& con : connections) {
      con->Query("RESET SESSION gpu_execution");
    }
    a.Query("RESET SESSION gpu_execution");
    a.Query("RESET GLOBAL gpu_execution");
  }

  duckdb::Connection& make_connection()
  {
    connections.push_back(
      std::make_unique<duckdb::Connection>(sirius::test::g_integration_env->database()));
    return *connections.back();
  }

  static bool enabled(duckdb::Connection& con)
  {
    duckdb::Value value;
    auto lookup = con.context->TryGetCurrentSetting("gpu_execution", value);
    REQUIRE(static_cast<bool>(lookup));
    REQUIRE_FALSE(value.IsNull());
    return value.GetValue<bool>();
  }

  static void check_routing(duckdb::Connection& con, bool gpu)
  {
    auto before = sirius::test::get_transparent_execution_stats(con);
    auto result = query(con, "SELECT 42::INTEGER");
    auto after  = sirius::test::get_transparent_execution_stats(con);
    REQUIRE(result->GetValue(0, 0).GetValue<int32_t>() == 42);
    sirius::test::require_transparent_execution_delta(before, after, gpu ? 1 : 0, 0, gpu ? 1 : 0);
  }
};

}  // namespace

TEST_CASE_METHOD(scope_fixture,
                 "gpu_execution unqualified SET affects only its connection",
                 "[integration][transparent][scope]")
{
  auto& c = make_connection();
  query(a, "SET gpu_execution = false");
  auto& b = make_connection();
  REQUIRE_FALSE(enabled(a));
  REQUIRE(enabled(b));
  REQUIRE(enabled(c));
  check_routing(a, false);
  check_routing(b, true);
  check_routing(c, true);
}

TEST_CASE_METHOD(scope_fixture,
                 "gpu_execution GLOBAL reaches new and existing connections without overrides",
                 "[integration][transparent][scope]")
{
  auto& c = make_connection();
  auto& d = make_connection();
  query(d, "SET SESSION gpu_execution = true");
  query(a, "SET GLOBAL gpu_execution = false");
  auto& b = make_connection();
  REQUIRE_FALSE(enabled(a));
  REQUIRE_FALSE(enabled(b));
  REQUIRE_FALSE(enabled(c));
  REQUIRE(enabled(d));
  check_routing(b, false);
  check_routing(c, false);
  check_routing(d, true);
}

TEST_CASE_METHOD(scope_fixture,
                 "gpu_execution RESET GLOBAL restores the default and retains session overrides",
                 "[integration][transparent][scope]")
{
  auto& c = make_connection();
  auto& d = make_connection();
  query(d, "SET SESSION gpu_execution = false");
  query(a, "SET GLOBAL gpu_execution = false");
  auto& b = make_connection();
  REQUIRE_FALSE(enabled(b));
  REQUIRE_FALSE(enabled(c));
  query(a, "RESET GLOBAL gpu_execution");
  REQUIRE(enabled(b));
  REQUIRE(enabled(c));
  REQUIRE_FALSE(enabled(d));
  check_routing(b, true);
  check_routing(c, true);
  check_routing(d, false);
}

TEST_CASE_METHOD(scope_fixture,
                 "gpu_execution RESET SESSION writes a session value equal to the default",
                 "[integration][transparent][scope]")
{
  auto& c = make_connection();
  auto& d = make_connection();
  query(d, "SET SESSION gpu_execution = false");
  query(a, "SET GLOBAL gpu_execution = false");
  query(d, "RESET SESSION gpu_execution");
  REQUIRE(enabled(d));
  REQUIRE_FALSE(enabled(c));
  query(a, "SET GLOBAL gpu_execution = true");
  REQUIRE(enabled(c));
  REQUIRE(enabled(d));
  query(a, "SET GLOBAL gpu_execution = false");
  REQUIRE_FALSE(enabled(c));
  REQUIRE(enabled(d));
  check_routing(c, false);
  check_routing(d, true);
}

TEST_CASE_METHOD(scope_fixture,
                 "gpu_execution unqualified RESET resets only the session to the default",
                 "[integration][transparent][scope]")
{
  auto& c = make_connection();
  auto& d = make_connection();
  query(d, "SET gpu_execution = false");
  query(a, "SET GLOBAL gpu_execution = false");
  query(d, "RESET gpu_execution");
  REQUIRE(enabled(d));
  REQUIRE_FALSE(enabled(c));
  query(a, "SET GLOBAL gpu_execution = true");
  query(a, "SET GLOBAL gpu_execution = false");
  REQUIRE(enabled(d));
  REQUIRE_FALSE(enabled(c));
  check_routing(d, true);
  check_routing(c, false);
}

TEST_CASE_METHOD(scope_fixture,
                 "gpu_execution settings and resets do not cross DatabaseInstances",
                 "[integration][transparent][scope]")
{
  duckdb::DuckDB other(nullptr);
  duckdb::Connection b(other);
  REQUIRE(enabled(b));
  for (auto const* sql : {"SET gpu_execution = false",
                          "SET GLOBAL gpu_execution = false",
                          "SET SESSION gpu_execution = false",
                          "RESET gpu_execution",
                          "RESET SESSION gpu_execution",
                          "RESET GLOBAL gpu_execution"}) {
    CAPTURE(sql);
    query(a, sql);
    REQUIRE(enabled(b));
  }
}
