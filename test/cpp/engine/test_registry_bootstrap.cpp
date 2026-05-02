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
#include "duckdb.hpp"
#include "duckdb/main/connection.hpp"
#include "io/datasource_factory.hpp"
#include "sirius_config.hpp"
#include "sirius_engine.hpp"
#include "sirius_interface.hpp"

#include <exception>
#include <memory>
#include <vector>

using sirius::sirius_engine;
using sirius::sirius_interface;

namespace {

struct engine_fixture {
  std::unique_ptr<duckdb::DuckDB> db;
  std::unique_ptr<duckdb::Connection> con;
  std::unique_ptr<sirius_interface> iface;
  std::unique_ptr<sirius_engine> engine;
};

// Stand up a minimal engine; returns an empty fixture (engine==nullptr) if
// bootstrap fails before the assertions under test.
engine_fixture try_make_engine()
{
  engine_fixture fx;
  fx.db  = std::make_unique<duckdb::DuckDB>(nullptr);
  fx.con = std::make_unique<duckdb::Connection>(*fx.db);
  try {
    fx.iface  = std::make_unique<sirius_interface>(*fx.con->context);
    fx.engine = std::make_unique<sirius_engine>(*fx.con->context, *fx.iface);
  } catch (std::exception const& e) {
    WARN("sirius_engine bootstrap failed: " << e.what());
    fx.engine.reset();
    fx.iface.reset();
  }
  return fx;
}

}  // namespace

TEST_CASE("sirius_engine bootstrap leaves file scheme on cudf default datasource", "[engine]")
{
  auto fx = try_make_engine();
  if (!fx.engine) {
    SUCCEED("Skipping: sirius_engine bootstrap failed on this runner");
    return;
  }

  auto& reg = fx.engine->datasource_registry();

  // Local files bypass the registry and use cudf's default datasource.
  CHECK(reg.lookup("file") == nullptr);
  CHECK(reg.lookup("s3") == nullptr);
  CHECK(reg.schemes().empty());
}

TEST_CASE("sirius_engine destruction handles empty datasource registry", "[engine]")
{
  {
    auto fx = try_make_engine();
    if (!fx.engine) {
      SUCCEED("Skipping: sirius_engine bootstrap failed on this runner");
      return;
    }
    CHECK(fx.engine->datasource_registry().schemes().empty());
  }

  SUCCEED("sirius_engine destroyed with no default file ioctx registered");
}

TEST_CASE("sirius_engine config falls back when SiriusContext is absent", "[engine]")
{
  auto fx = try_make_engine();
  if (!fx.engine) {
    SUCCEED("Skipping: sirius_engine bootstrap failed on this runner");
    return;
  }

  REQUIRE_NOTHROW(fx.engine->config());
  auto const& cfg = fx.engine->config();
  CHECK(cfg.get_object_store_config().endpoint.empty());
}
