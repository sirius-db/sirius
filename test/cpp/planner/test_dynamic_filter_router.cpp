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
 * @file test_dynamic_filter_router.cpp
 * @brief Tests for sirius_physical_plan_generator::get_or_create_dynamic_filter_channel —
 *        the lookup-or-create map that pairs producer joins and consumer scans via the
 *        duckdb::DynamicTableFilterSet pointer identity.
 */

#include "op/sirius_dynamic_filter.hpp"
#include "planner/sirius_physical_plan_generator.hpp"

#include <catch.hpp>
#include <duckdb.hpp>
#include <duckdb/planner/table_filter.hpp>

using sirius::op::sirius_dynamic_filter_set;
using sirius::planner::sirius_physical_plan_generator;

namespace {

/// Stand-in fixture: a real DuckDB connection so we can construct a plan generator
/// (which requires a ClientContext) without standing up the whole query pipeline.
struct router_fixture {
  router_fixture() : db(nullptr), con(db) {}
  duckdb::DuckDB db;
  duckdb::Connection con;
};

}  // namespace

TEST_CASE_METHOD(router_fixture,
                 "get_or_create_dynamic_filter_channel returns nullptr for a null key",
                 "[dynamic_filter][router]")
{
  sirius_physical_plan_generator gen(*con.context);
  REQUIRE(gen.get_or_create_dynamic_filter_channel(nullptr) == nullptr);
  REQUIRE(gen.dynamic_filter_channels.empty());
}

TEST_CASE_METHOD(router_fixture,
                 "get_or_create_dynamic_filter_channel creates a channel for a new key",
                 "[dynamic_filter][router]")
{
  sirius_physical_plan_generator gen(*con.context);
  duckdb::DynamicTableFilterSet key;

  auto channel = gen.get_or_create_dynamic_filter_channel(&key);

  REQUIRE(channel != nullptr);
  REQUIRE(channel->empty());
  REQUIRE(gen.dynamic_filter_channels.size() == 1);
}

TEST_CASE_METHOD(router_fixture,
                 "get_or_create_dynamic_filter_channel is idempotent for the same key",
                 "[dynamic_filter][router]")
{
  sirius_physical_plan_generator gen(*con.context);
  duckdb::DynamicTableFilterSet key;

  auto first  = gen.get_or_create_dynamic_filter_channel(&key);
  auto second = gen.get_or_create_dynamic_filter_channel(&key);

  REQUIRE(first.get() == second.get());
  REQUIRE(gen.dynamic_filter_channels.size() == 1);
}

TEST_CASE_METHOD(router_fixture,
                 "get_or_create_dynamic_filter_channel mints distinct channels for distinct keys",
                 "[dynamic_filter][router]")
{
  sirius_physical_plan_generator gen(*con.context);
  duckdb::DynamicTableFilterSet key_a;
  duckdb::DynamicTableFilterSet key_b;

  auto channel_a = gen.get_or_create_dynamic_filter_channel(&key_a);
  auto channel_b = gen.get_or_create_dynamic_filter_channel(&key_b);

  REQUIRE(channel_a != nullptr);
  REQUIRE(channel_b != nullptr);
  REQUIRE(channel_a.get() != channel_b.get());
  REQUIRE(gen.dynamic_filter_channels.size() == 2);
}

TEST_CASE_METHOD(router_fixture,
                 "channels handed out by the router survive after the generator is destroyed",
                 "[dynamic_filter][router]")
{
  std::shared_ptr<sirius_dynamic_filter_set> channel;
  duckdb::DynamicTableFilterSet key;
  {
    sirius_physical_plan_generator gen(*con.context);
    channel = gen.get_or_create_dynamic_filter_channel(&key);
  }
  REQUIRE(channel != nullptr);
  REQUIRE(channel->empty());
}
