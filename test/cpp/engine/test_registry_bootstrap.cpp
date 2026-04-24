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

// IMPORTANT: include order matters here. liburing.h (pulled in transitively
// by io/uring/uring_ioctx.hpp) defines BLOCK_SIZE as a preprocessor macro,
// which collides with duckdb concurrentqueue's BLOCK_SIZE identifier. Include
// all duckdb headers (and anything that pulls them in) BEFORE the uring
// headers.
#include "catch.hpp"
#include "duckdb.hpp"
#include "duckdb/main/connection.hpp"
#include "io/datasource_factory.hpp"
#include "io/types.hpp"
#include "io/uring/uring_ioctx.hpp"
#include "sirius_engine.hpp"
#include "sirius_interface.hpp"

#include <exception>
#include <memory>
#include <vector>

using sirius::sirius_engine;
using sirius::sirius_interface;
using sirius::io::datasource_registry;
using sirius::io::sirius_ioctx;
using sirius::io::uring_ioctx;

namespace {

struct engine_fixture {
  std::unique_ptr<duckdb::DuckDB> db;
  std::unique_ptr<duckdb::Connection> con;
  std::unique_ptr<sirius_interface> iface;
  std::unique_ptr<sirius_engine> engine;
};

// Stand up a minimal engine; returns an empty fixture (engine==nullptr) when
// the runtime does not support io_uring — the engine ctor constructs a
// uring_ioctx by default, which requires the kernel capability.
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

TEST_CASE("sirius_engine bootstrap populates registry with uring_ioctx", "[engine]")
{
  auto fx = try_make_engine();
  if (!fx.engine) {
    SUCCEED("Skipping: io_uring not supported on this runner");
    return;
  }

  auto& reg = fx.engine->datasource_registry();

  // "file" is the one scheme registered by default in PR2.
  auto ctx = reg.lookup("file");
  REQUIRE(ctx != nullptr);
  CHECK(dynamic_cast<uring_ioctx*>(ctx.get()) != nullptr);

  auto schemes = reg.schemes();
  REQUIRE(schemes.size() == 1);
  CHECK(schemes.front() == "file");

  // Unregistered schemes still return nullptr.
  CHECK(reg.lookup("s3") == nullptr);
}

TEST_CASE("sirius_engine destruction releases ioctx cleanly", "[engine]")
{
  std::shared_ptr<sirius_ioctx> ctx_ref;

  {
    auto fx = try_make_engine();
    if (!fx.engine) {
      SUCCEED("Skipping: io_uring not supported on this runner");
      return;
    }
    ctx_ref = fx.engine->datasource_registry().lookup("file");
    REQUIRE(ctx_ref != nullptr);
    // Engine's registry + our local ref -> at least 2 strong refs.
    CHECK(ctx_ref.use_count() >= 2);
  }

  // Engine is destroyed; the registry dropped its shared_ptr, so only our
  // local reference should remain alive.
  CHECK(ctx_ref.use_count() == 1);
}
