/*
 * Copyright 2025, Sirius Contributors.
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

#include "catch.hpp"
#include "memory/common.hpp"
#include "sirius_context.hpp"

#include <duckdb.hpp>
#include <duckdb/execution/execution_context.hpp>
#include <libconfig.h++>

#include <cstdlib>  // for setenv/putenv
#include <filesystem>
#include <fstream>
#include <iostream>
#include <source_location>
#include <string>

namespace fs = std::filesystem;

struct finally {
  std::function<void()> func;
  ~finally()
  {
    if (func) { func(); }
  }
};

TEST_CASE("Sirius configuration loading from file with configurator", "[sirius][context]")
{
  finally cleanup_env{[]() { unsetenv("SIRIUS_CONFIG_FILE"); }};

  std::source_location loc = std::source_location::current();
  fs::path cfg             = fs::path(loc.file_name()).parent_path() / "data" / "configurator.cfg";

  setenv("SIRIUS_CONFIG_FILE", cfg.string().c_str(), 1);

  duckdb::DuckDB db(nullptr);
  duckdb::Connection con(db);

  // get client context from con
  auto& client_ctx = *con.context;
  // get registered sirius context
  auto sirius_ctx = client_ctx.registered_state->Get<duckdb::SiriusContext>("sirius_state");

  REQUIRE(sirius_ctx != nullptr);

  auto& manager = sirius_ctx->get_memory_manager();
  REQUIRE(manager.get_all_memory_spaces().size() == 3);
  REQUIRE(manager.get_memory_spaces_for_tier(cucascade::memory::Tier::GPU).size() == 1);
  REQUIRE(manager.get_memory_spaces_for_tier(cucascade::memory::Tier::HOST).size() == 1);
  REQUIRE(manager.get_memory_spaces_for_tier(cucascade::memory::Tier::DISK).size() == 1);
}

TEST_CASE("Sirius configuration loading from file with spaces", "[sirius][context]")
{
  finally cleanup_env{[]() { unsetenv("SIRIUS_CONFIG_FILE"); }};

  std::source_location loc = std::source_location::current();
  fs::path cfg             = fs::path(loc.file_name()).parent_path() / "data" / "spaces.cfg";

  setenv("SIRIUS_CONFIG_FILE", cfg.string().c_str(), 1);

  duckdb::DuckDB db(nullptr);
  duckdb::Connection con(db);

  auto& client_ctx = *con.context;
  auto sirius_ctx  = client_ctx.registered_state->Get<duckdb::SiriusContext>("sirius_state");

  REQUIRE(sirius_ctx != nullptr);

  auto& manager = sirius_ctx->get_memory_manager();
  REQUIRE(manager.get_all_memory_spaces().size() == 4);
  REQUIRE(manager.get_memory_spaces_for_tier(cucascade::memory::Tier::GPU).size() == 1);
  REQUIRE(manager.get_memory_spaces_for_tier(cucascade::memory::Tier::HOST).size() == 1);
  REQUIRE(manager.get_memory_spaces_for_tier(cucascade::memory::Tier::DISK).size() == 2);
  REQUIRE(manager.get_all_memory_spaces().size() == 4);
}
